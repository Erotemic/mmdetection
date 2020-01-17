import torch
import ubelt as ub


class SamplingResult(ub.NiceRepr):
    """

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print('self = {}'.format(self))
        self = <SamplingResult({
            'neg_bboxes': tensor([[0.0304, 0.0786, 0.9552, 0.6478],
                                  [0.1156, 0.0271, 0.3986, 0.2002],
                                  [0.0231, 0.0357, 0.1798, 0.7272],
                                  [0.5397, 0.5367, 0.8913, 0.6369],
                                  [0.1174, 0.1481, 0.5035, 0.6931],
                                  [0.7211, 0.3756, 0.7253, 0.3812]]),
            'neg_inds': tensor([0, 2, 3, 4, 6, 7]),
            'num_gts': 5,
            'pos_assigned_gt_inds': tensor([4, 4, 0, 1, 4]),
            'pos_bboxes': tensor([[0.5009, 0.5779, 0.8551, 0.8269],
                                  [0.6203, 0.3180, 0.6829, 0.4012],
                                  [0.4141, 0.2988, 0.4976, 0.9158],
                                  [0.5334, 0.4964, 0.9138, 0.7918],
                                  [0.2286, 0.1082, 0.5891, 0.7869]]),
            'pos_inds': tensor([ 1,  5,  8,  9, 10]),
            'pos_is_gt': tensor([0, 0, 0, 0, 0], dtype=torch.uint8),
        })>

    """
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """
        Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print('self = {}'.format(self.to(None)))
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print('self = {}'.format(self.to(0)))
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        return ub.repr2(self.info, nl=1)

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
    def random(cls, rng=None, **kwargs):
        """
        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print('{}'.format(ub.repr2(self.__dict__, nl=1)))

            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random(num_gts=0, num_preds=0)
            >>> print('self = {}'.format(self))

            >>> self = SamplingResult.random(num_gts=0, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=3, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=0, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=7, num_preds=7)
            >>> print('self = {}'.format(self))

            >>> self = SamplingResult.random(num_gts=7, num_preds=64)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

            >>> self = SamplingResult.random(num_gts=24, num_preds=3)
            >>> print('{}'.format(ub.repr2(self.info, nl=1)))

        Ignore:
            for i in range(1000):
                self = SamplingResult.random(rng=i)

            self = SamplingResult.random(rng=11)
        """
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(rng)

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
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
                                add_gt_as_proposals=add_gt_as_proposals,
                                rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self
