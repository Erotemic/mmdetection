"""
Autogenerate a small dataset for demo and testing purposes

Tests the trainer as follows
tools/demo_dataset/demo_dataset.mscoco.json

Requirements:
    pip install ndsampler


Notes:
    python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
        checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
        --show
"""
from os.path import join, realpath, relpath, dirname


def main():
    import ndsampler
    import ubelt as ub

    def _ensure_empty_image(dset):
        # ensure that at least X% of images have no truth annotations
        # target_frac = 0.05
        target_frac = 0.20
        gid_to_nannots = ub.map_vals(len, dset.gid_to_aids)
        import numpy as np
        gids = np.array(list(gid_to_nannots.keys()))
        n_annots = np.array(list(gid_to_nannots.values()))
        n_empty = (n_annots == 0).sum()
        percent_empty = n_empty / len(n_annots)
        print('old percent_empty = {!r}'.format(percent_empty))
        num_need_total = int(target_frac * len(n_annots))
        num_need = (num_need_total - n_empty)

        if num_need > 0:
            idxs = np.where(n_annots > 0)[0][:num_need]
            remove_anns_in_gids = gids[idxs]

            remove_aids = list(ub.flatten(ub.take(dset.gid_to_aids, remove_anns_in_gids)))
            dset.remove_annotations(remove_aids)

        gid_to_nannots = ub.map_vals(len, dset.gid_to_aids)
        import numpy as np
        gids = np.array(list(gid_to_nannots.keys()))
        n_annots = np.array(list(gid_to_nannots.values()))
        n_empty = (n_annots == 0).sum()
        percent_empty = n_empty / len(n_annots)
        percent_empty = n_empty / len(n_annots)
        print('new percent_empty = {!r}'.format(percent_empty))

    # Create a random coco dataset
    dpath = ub.ensuredir(realpath('./demodata/shapes'))
    # ub.delete(dpath)

    annot_dpath = ub.ensuredir((dpath, 'annotations'))

    train_dset = ndsampler.CocoDataset.demo(key='shapes1024', dpath=dpath,
                                            newstyle=False)
    vali_dset = ndsampler.CocoDataset.demo(key='shapes256', dpath=dpath,
                                           newstyle=False)
    test_dset = ndsampler.CocoDataset.demo(key='shapes128', dpath=dpath,
                                           newstyle=False)
    _ensure_empty_image(train_dset)
    _ensure_empty_image(vali_dset)
    _ensure_empty_image(test_dset)

    train_dset.fpath = join(annot_dpath, 'instances_train.mscoco.json')
    train_dset.dump(train_dset.fpath, newlines=True)

    vali_dset.fpath = join(annot_dpath, 'instances_vali.mscoco.json')
    vali_dset.dump(vali_dset.fpath, newlines=True)

    test_dset.fpath = join(annot_dpath, 'instances_test.mscoco.json')
    test_dset.dump(test_dset.fpath, newlines=True)

    base_config_fpath = ub.expandpath(
        '~/code/mmdetection/configs/cascade_rcnn_r50_fpn_1x.py')

    with open(base_config_fpath, 'r') as file:
        text = file.read()

    new_data_root = dpath
    text = text.replace('data/coco/', new_data_root + '/')
    text = text.replace('annotations/instances_train2017.json', relpath(train_dset.fpath, new_data_root))
    text = text.replace('annotations/instances_val2017.json', relpath(vali_dset.fpath, new_data_root))
    text = text.replace('annotations/instances_val2017.json', relpath(test_dset.fpath, new_data_root))

    print('--train_dataset={} \\'.format(train_dset.fpath))
    print('--vali_dataset={} \\'.format(vali_dset.fpath))

    text = text.replace('train2017/', relpath(dirname(test_dset.imgs[1]['file_name']), new_data_root))
    text = text.replace('val2017/', relpath(dirname(vali_dset.imgs[1]['file_name']), new_data_root))
    text = text.replace('val2017/', relpath(dirname(test_dset.imgs[1]['file_name']), new_data_root))

    text = text.replace('workers_per_gpu=2,', 'workers_per_gpu=2, filter_empty_gt=False,')

    text = text.replace('./work_dirs/cascade_rcnn_r50_fpn_1x', './work_dirs/hack_test')

    hack_config_fpath = ub.augpath(base_config_fpath, base='hack_config')
    with open(hack_config_fpath, 'w') as file:
        file.write(text)

    # see also: ~/code/mmdetection/docs/GETTING_STARTED.md
    print(ub.codeblock(
        '''
        python tools/train.py --gpus 2 {hack_config_fpath}


        python tools/train.py --gpus 2 configs/hack_config.py

        ''').format(**locals()))


if __name__ == '__main__':
    """
    python ~/code/mmdetection/tools/make_demo_dataset.py

        python -m bioharn.detect_fit \
            --nice=test-multi-gpu \
            --train_dataset=/home/joncrall/code/mmdetection/demodata/shapes/annotations/instances_train.mscoco.json \
            --vali_dataset=/home/joncrall/code/mmdetection/demodata/shapes/annotations/instances_vali.mscoco.json \
            --workdir=~/work/test \
            --schedule=ReduceLROnPlateau-p2-c2 \
            --augment=complex \
            --init=noop \
            --arch=cascade \
            --use_disparity=False \
            --optim=sgd --lr=3e-3 \
            --input_dims=window \
            --window_dims=512,512 \
            --window_overlap=0.3 \
            --multiscale=True \
            --normalize_inputs=False \
            --workers=4 --xpu=1,0 --batch_size=8 --bstep=4
    """
    main()
