import os
import pdb
import numpy as np
import pickle
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

from utils.common_utils import AverageMeter, accuracy


class Trainer():
    def __init__(self, args, model, logger, criterion, criterion_person, optimizer,
                 train_loader=None, test_loader=None):

        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_person = criterion_person
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def KLcriterion(self,p,q):
        loss = F.kl_div(torch.log_softmax(q, dim=1), torch.softmax(p, dim=1), reduction='mean')
        return loss
    def train(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        topi = AverageMeter()
        topc = AverageMeter()
        topgi = AverageMeter()
        topT = AverageMeter()
        constras_loss = AverageMeter()


        batch_start_time = time.time()

        self.model.train()
        self.criterion.train()
        self.criterion_person.train()

        for i, batch_data in enumerate(self.train_loader):
            (joint_feats, targets_thisbatch,
             video_thisbatch, clip_thisbatch,
             person_labels, ball_feats) = batch_data

            data_time.update(time.time() - batch_start_time)

            self.optimizer.zero_grad()

            # normalize the prototypes
            with torch.no_grad():
                w = self.model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.module.prototypes.weight.copy_(w)
                # delete module
            joint_feats = joint_feats.cuda()
            ball_feats = ball_feats.cuda()
            # model forward pass
            pred_logits_thisbatch, pred_logits_person, scores,  Pdata = self.model(
                joint_feats, ball_feats)

            scores_c = scores[0]
            scores_i = scores[1]
            scores_g = scores[2]
            scores_m = scores[3]

            with torch.no_grad():
                q_c = self.sinkhorn(scores_c, nmb_iters=self.args.sinkhorn_iterations)
                q_i = self.sinkhorn(scores_i, nmb_iters=self.args.sinkhorn_iterations)
                q_g = self.sinkhorn(scores_g, nmb_iters=self.args.sinkhorn_iterations)
                q_m = self.sinkhorn(scores_m, nmb_iters=self.args.sinkhorn_iterations)


            p_m = scores_m / self.args.temperature
            p_c = scores_c / self.args.temperature
            p_i = scores_i / self.args.temperature
            p_g = scores_g / self.args.temperature

            contrastive_clustering_loss = self.args.loss_coe_constrastive_clustering * (
                    self.swap_prediction(p_c, p_i, q_c, q_i) +
                    self.swap_prediction(p_m, p_c, q_m, q_c) +
                    self.swap_prediction(p_m, p_i, q_m, q_i) +
                    self.swap_prediction(p_m, p_g, q_m, q_g) +
                    self.swap_prediction(p_i, p_g, q_i, q_g) +
                    self.swap_prediction(p_c, p_g, q_c, q_g)
            ) / 6.0  # 6 pairs of views

            constras_loss.update(contrastive_clustering_loss.data.item(), len(targets_thisbatch))



            # measure accuracy and record loss 
            targets_thisbatch = targets_thisbatch.to(pred_logits_thisbatch[0][0].device)
            person_labels = person_labels.flatten(0, 1).to(pred_logits_thisbatch[0][0].device)
            loss_thisbatch, prec1, prec3,  prec1i, prec1gi, prec1c, preTotal1 = self.loss_acc_compute(
                pred_logits_thisbatch, targets_thisbatch, pred_logits_person, person_labels)


            loss_thisbatch += contrastive_clustering_loss

            loss.update(loss_thisbatch.data.item(), len(targets_thisbatch))

            top1.update(prec1.item(), len(targets_thisbatch))
            topi.update(prec1i.item(), len(targets_thisbatch))
            topc.update(prec1c.item(), len(targets_thisbatch))
            topgi.update(prec1gi.item(), len(targets_thisbatch))
            topT.update(preTotal1.item(), len(targets_thisbatch))
            top3.update(prec3.item(), len(targets_thisbatch))


            loss_thisbatch.backward()
            self.optimizer.step()

            # finish
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            self.logger.info('Train [e{0:02d}][{1}/{2}] '
                             'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                             'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                             'Loss {loss.val:.4f}({loss.avg:.4f}) '
                             'Loss Constrastive {constras_loss.val:.4f}({constras_loss.avg:.4f}) '
                             'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                             'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                             'Topi {topi.val:.4f}({topi.avg:.4f}) '
                             'Topc {topc.val:.4f}({topc.avg:.4f}) '
                             'Topgi {topgi.val:.4f}({topgi.avg:.4f}) '
                             'TopT {topT.val:.4f}({topT.avg:.4f}) '.format(
                epoch, i + 1, len(self.train_loader), batch_time=batch_time,
                data_time=data_time,
                loss=loss, constras_loss=constras_loss, top1=top1, top3=top3,
                topi=topi, topgi=topgi, topc=topc, topT=topT))


    @torch.no_grad()
    def test(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        topi = AverageMeter()
        topc = AverageMeter()
        topgi = AverageMeter()
        topT = AverageMeter()

        batch_start_time = time.time()

        self.model.eval()
        self.criterion.eval()
        self.criterion_person.eval()

        for i, batch_data in enumerate(self.test_loader):
            (joint_feats, targets_thisbatch,
             video_thisbatch, clip_thisbatch,
             person_labels, ball_feats) = batch_data

            data_time.update(time.time() - batch_start_time)

            # normalize the prototypes
            with torch.no_grad():
                # delete module
                w = self.model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.module.prototypes.weight.copy_(w)

            joint_feats = joint_feats.cuda()
            ball_feats = ball_feats.cuda()
            # model forward
            pred_logits_thisbatch, pred_logits_person, _,  Pdata = self.model(joint_feats,
                                                                                          ball_feats)

            # measure accuracy and record loss 
            targets_thisbatch = targets_thisbatch.to(pred_logits_thisbatch[0][0].device)
            person_labels = person_labels.flatten(0, 1).to(pred_logits_thisbatch[0][0].device)

            loss_thisbatch, prec1, prec3,  prec1i, prec1gi, prec1c, preTotal1 = self.loss_acc_compute(
                pred_logits_thisbatch, targets_thisbatch, pred_logits_person, person_labels)



            loss.update(loss_thisbatch.data.item(), len(targets_thisbatch))

            top1.update(prec1.item(), len(targets_thisbatch))
            top3.update(prec3.item(), len(targets_thisbatch))
            topi.update(prec1i.item(), len(targets_thisbatch))
            topc.update(prec1c.item(), len(targets_thisbatch))
            topgi.update(prec1gi.item(), len(targets_thisbatch))
            topT.update(preTotal1.item(), len(targets_thisbatch))

            # finish
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            self.logger.info('Test [e{0:02d}][{1}/{2}] '
                             'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                             'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                             'Loss {loss.val:.4f}({loss.avg:.4f}) '
                             'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                             'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                             'Topi {topi.val:.4f}({topi.avg:.4f}) '
                             'Topc {topc.val:.4f}({topc.avg:.4f}) '
                             'Topgi {topgi.val:.4f}({topgi.avg:.4f}) '
                             'TopT {topT.val:.4f}({topT.avg:.4f}) '
                             .format(
                epoch, i + 1, len(self.test_loader), batch_time=batch_time,
                data_time=data_time,
                loss=loss, top1=top1, top3=top3,
                topi=topi, topgi=topgi, topc=topc, topT=topT))
        return top1.avg, top3.avg, loss.avg, topi.avg, topc.avg, topgi.avg,topT.avg

    def loss_acc_compute(
            self, pred_logits_thisbatch, targets_thisbatch,
            pred_logits_person=None, person_labels=None):

        loss_thisbatch = 0

        for l in range(self.args.trans_layers):
            if l == self.args.trans_layers - 1:  # if last layer
                loss_thisbatch += self.args.loss_coe_last_TNT * (
                        self.args.loss_coe_relation * self.criterion(pred_logits_thisbatch[l][0], targets_thisbatch) +
                        self.args.loss_coe_inter * self.criterion(pred_logits_thisbatch[l][1], targets_thisbatch) +
                        self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][2], targets_thisbatch) +
                        self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][3], targets_thisbatch) +
                        self.args.loss_coe_person * self.criterion_person(pred_logits_person[l], person_labels) )

                prec1, prec3 = accuracy(pred_logits_thisbatch[l][2], targets_thisbatch, topk=(1, 3))
                prec1i, prec3i = accuracy(pred_logits_thisbatch[l][1], targets_thisbatch, topk=(1, 3))
                prec1c, prec3c = accuracy(pred_logits_thisbatch[l][0], targets_thisbatch, topk=(1, 3))
                prec1gi, prec3gi = accuracy(pred_logits_thisbatch[l][3], targets_thisbatch, topk=(1, 3))
                total = pred_logits_thisbatch[l][0] + pred_logits_thisbatch[l][1] + pred_logits_thisbatch[l][-1]
                preTotal1, preTotal3 = accuracy(total, targets_thisbatch, topk=(1, 3))

            else:  # not last layer
                loss_thisbatch += (
                        self.args.loss_coe_relation * self.criterion(pred_logits_thisbatch[l][0], targets_thisbatch) +
                        self.args.loss_coe_inter * self.criterion(pred_logits_thisbatch[l][1], targets_thisbatch) +
                        self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][2], targets_thisbatch) +
                        self.args.loss_coe_group * self.criterion(pred_logits_thisbatch[l][3], targets_thisbatch) +
                        self.args.loss_coe_person * self.criterion_person(pred_logits_person[l], person_labels))

        return loss_thisbatch, prec1, prec3, prec1i, prec1gi, prec1c, preTotal1


    def sinkhorn(self, scores, epsilon=0.05, nmb_iters=3):
        with torch.no_grad():
            Q = torch.exp(scores / epsilon).t()
            K, B = Q.shape

            sum_Q = torch.sum(Q)
            Q /= sum_Q

            if len(self.args.gpu) > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def swap_prediction(self, p_t, p_s, q_t, q_s):
        loss = - 0.5 * (
                torch.mean(
                    torch.sum(
                        q_t * F.log_softmax(p_s, dim=1), dim=1)
                ) + torch.mean(torch.sum(q_s * F.log_softmax(p_t, dim=1), dim=1)))
        return loss
