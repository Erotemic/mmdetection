import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        FIX = False
        if FIX and gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    @property
    def info(self):
        """
        Returns a dictionary of info about the object
        """
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, **kwargs):
        """
        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print('{}'.format(ub.repr2(self.__dict__, nl=1)))

            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random(num_gts=0, num_preds=0)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=0, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=3, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=0, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=7, num_preds=7)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=7, num_preds=64)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=24, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))
        """
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox import demodata

        rng = demodata.ensure_rng(kwargs.pop('rng', None))

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(**kwargs)

        # Note we could just compute an assignment
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        gt_bboxes = gt_bboxes.squeeze()
        bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # make probabalistic?

        sampler = RandomSampler(num, pos_fraction, neg_pos_ubo=neg_pos_ub,
                                add_gt_as_proposals=add_gt_as_proposals)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self
