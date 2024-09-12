import argparse
import datetime as dt
from pathlib import Path
import sys
import captum.attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from captum.attr import visualization as viz
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from torchvision import models

sys.path.append('/home/mw/Repos/itc-main-repo/Ming')

from pytorch.check_cuda import get_device


def save_attr_result(attributions, transformed_img, img_path, method_name, pred_label, pred_score, cmap=None):
    if cmap is None:
        # https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, 'blue'),
                                                          (0.5, 'yellow'),
                                                          (1, 'red')], N=256, gamma=0.8)
        default_cmap = cm.get_cmap("jet")
        sal_cmap = cm.get_cmap("bwr")
    img_basename = Path(img_path).stem  # serve as prefix
    img_folder = Path(img_path).parent  # the result will be saved to the same folder as the submitted image
    plot_title = "Method:{} Predicted:{}({:.2f})".format(method_name, pred_label, pred_score)

    # print(attributions.shape,transformed_img.shape)
    # first save a side by side version
    _ = viz.visualize_image_attr_multiple(attr=np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          original_image=np.transpose(transformed_img.squeeze().cpu().detach().numpy(),
                                                                      (1, 2, 0)),
                                          methods=["original_image", "heat_map"],
                                          signs=["all", "all"],
                                          show_colorbar=True,
                                          titles=["{}\nOriginal image ".format(plot_title), "Attribution map"],
                                          cmap=sal_cmap,
                                          fig_size=(8, 8), use_pyplot=False
                                          )

    # save the image
    fname = "{}/{}_{}_{}_{:.3f}_side.png".format(img_folder,
                                                 img_basename, method_name, pred_label, pred_score)
    _[0].savefig(fname, dpi=400)
    # an overlay version
    _ = viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='blended_heat_map', alpha_overlay=0.5, cmap=default_cmap, show_colorbar=True,
                                 title=plot_title, fig_size=(8, 8), use_pyplot=False)

    # save the image
    fname = "{}/{}_{}_{}_{:.3f}_overlay.png".format(img_folder,
                                                    img_basename, method_name, pred_label, pred_score)
    _[0].savefig(fname, dpi=400)

    print("Attribution results saved to images in the same folder.")
    return _


def generate_attr_maps(model, img_path, classes, transform_config, **kwargs):
    # attribution methods that can be called
    # perturbation based
    occ = kwargs.get('occlusionSensitivity', None)  # fixed square mask
    fa = kwargs.get('featureAblation', None)  # supplied mask
    fa_mask_path = kwargs.get('featureMask', None)  # for feature ablation
    # gradient based
    # pass sanity check: https://arxiv.org/pdf/1810.03292.pdf
    vanilla_backprop = kwargs.get('saliency', None)
    grad_cam = kwargs.get('gradCAM', None)
    grad_cam_layer = kwargs.get('gradCAMlayer', None)
    # failed sanity check, ~edge detector
    guided_backprop = kwargs.get('guidedBackProp', None)
    guided_grad_cam = kwargs.get('guidedGradCAM', None)
    deeplift = kwargs.get('deepLift', None)  # comparable with Integrated Gradients but faster
    ixg = kwargs.get('inputXGradient', None)
    ig = kwargs.get('integratedGradient', None)
    # not tested grad based
    # note guided backprop is combining the vanilla back prop with the trick in deconv
    deconv = kwargs.get('deconvolution', None)

    # noise tunnel has to be paired with one of the methods above , nt(saliency)~smoothGrad
    nt = kwargs.get('noiseTunnel', None)  # noise tunnel produces smoother output but higher memory consumption

    # use cuda or cpu
    device = get_device()

    # prepare image
    img = Image.open(img_path)
    img = img.convert('RGB')  # force 1 channel to 3 channels
    if albumentation_aug is not None:
        img = np.array(img)

        data = {"image": img}
        transformed_img = data_transforms[transform_config]['val'](**data)  # val no data augmentation
        transformed_img = transformed_img["image"].unsqueeze(0).to(device)

    else:
        transformed_img = data_transforms[transform_config]['val'](img)  # val no data augmentation
        transformed_img = transformed_img.unsqueeze(0).to(device)  # add dim for batch size
    # classify
    output = model(transformed_img)  # logit
    output = F.softmax(output, dim=1)  # prob
    pred_score, pred_label_idx = torch.topk(output, 1)
    prediction_score = pred_score.squeeze().item()  # tensor to float
    # squeeze back
    pred_label_idx.squeeze_()
    predicted_label = classes[pred_label_idx.item()]
    print('{}: Predicted:{} [{}]({:.3f})'.format(img_path, predicted_label, pred_label_idx,
                                                 prediction_score))
    # attribution maps below

    if occ is not None:
        # ***************************************Occlusion********************************************
        print("Attribution with occlusion sensitivity")
        occlusion = captum.attr.Occlusion(model)

        # Save time starting the attribution computation
        occ_start = dt.datetime.now()
        # hard-coded sliding window size and stride! ideally ~ the feature size ; memory use scales with window size
        for size in [8, 16, 24, 32, 40]:
            print("Sliding with window size {}".format(size))
            method = "occlusion_{}".format(size)

            attribution = occlusion.attribute(transformed_img,
                                              strides=(1),
                                              target=pred_label_idx,
                                              sliding_window_shapes=(3, size, size),
                                              baselines=0, perturbations_per_eval=1)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        delta = dt.datetime.now() - occ_start
        print("Time required for occlusion sensitivity:{}".format(delta))

    if fa is not None and fa_mask_path is not None:
        # ***************************************feature ablation********************************************
        print("Attribution with feature ablation")

        mask_img = Image.open(fa_mask_path)

        # convert the mask image to a numpy array (channels, height, width)
        feature_mask = np.array(mask_img)
        # count features
        features = np.unique(feature_mask)
        print("number of features plus background :{}".format(features.shape))
        # Captum expects consecutive group ids 0,1,2,3,4,..
        # this block for converting non-consecutive pixel values
        for idx, feature in np.ndenumerate(features):
            feature_mask[feature_mask == feature] = idx
        # convert the mask in array to tensor:
        if albumentation_aug is not None:
            data = {"image": feature_mask}
            feature_mask_tensor = data_transforms[transform_config]['val'](**data)  # val no data augmentation
            feature_mask_tensor = feature_mask_tensor["image"].unsqueeze(0).to(device)
        else:
            feature_mask_tensor = data_transforms[transform_config]['val'](feature_mask)  # val no data augmentation
            feature_mask_tensor = feature_mask_tensor.unsqueeze(0).to(device)  #add batch dimension

        ablator = captum.attr.FeatureAblation(model)
        # Save time starting the attribution computation
        ablator_start = dt.datetime.now()
        # hard-coded sliding window size and stride! ideally ~ the feature size ; memory use scales with window size
        attributions = ablator.attribute(transformed_img,
                                         target=pred_label_idx,
                                         feature_mask=feature_mask_tensor,
                                         baselines=0, perturbations_per_eval=1)
        delta = dt.datetime.now() - ablator_start

        print("Time required for feature ablation:{}".format(delta))

        method = "feaAbl"
        save_attr_result(attributions, transformed_img, img_path, method, predicted_label, prediction_score)

    # ***************************************************grad based methods*********************************************

    if vanilla_backprop is not None:
        #  ************************************integrated gradient*****************************************
        print("Attribution with vanilla backprop/saliency")

        sal = captum.attr.Saliency(model)
        method = "vanillaBackProp"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(sal)
            method = "salWNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM !
            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',stdevs=0.5,
                                                 target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            attribution = sal.attribute(transformed_img, target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if grad_cam is not None and grad_cam_layer is not None:
        #  ************************************gradCAM*****************************************
        print("Attribution with gradCAM")

        gc_layer = None
        # find the layer as torch.nn.Module object
        cur_conv = None
        for name, module in model.named_modules():
            if grad_cam_layer is None or name in grad_cam_layer:
                gc_layer = module
            if type(module) == nn.Conv2d:
                cur_conv = module
        if gc_layer is None:
            print("gradCAM: target layer not found, take the final conv layer")
            gc_layer = cur_conv
        print("layer for gradCAM:{}".format(gc_layer))
        layer_gra_cam = captum.attr.LayerGradCam(model, gc_layer)
        # Note in gradCAM paper relu==True
        attribution = layer_gra_cam.attribute(transformed_img, target=pred_label_idx,
                                              relu_attributions=True)

        # GradCAM attributions are often upsampled and viewed as a
        # mask to the input, since the convolutional layer output
        # spatially matches the original input image.
        # This can be done with LayerAttribution's interpolate method.
        # gradCAM as interpolated is coarse representation,noise tunnel not neccessary
        # transformed_img.shape=(count, channels, height, width)
        original_size = (transformed_img.shape[2], transformed_img.shape[3])
        attribution = captum.attr.LayerAttribution.interpolate(attribution, original_size, interpolate_mode="bilinear")
        attribution = torch.cat([attribution, attribution, attribution], dim=1)
        method = "gradCAM"
        save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if guided_backprop is not None:
        #  ************************************guided backProp *****************************************
        print("Attribution with guided backprop")

        gbp = captum.attr.GuidedBackprop(model)
        method = "GBP"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(gbp)
            method = "GBPwNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM ! n_sample/n_steps 10/10 already gives out of memory error for 8G GPU
            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            attribution = gbp.attribute(transformed_img, target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if guided_grad_cam is not None and grad_cam_layer is not None:
        #  ************************************guided grad cam*****************************************
        print("Attribution with guided gradCAM")

        gc_layer = None
        # find the layer as torch.nn.Module object
        cur_conv = None
        for name, module in model.named_modules():
            if grad_cam_layer is None or name in grad_cam_layer:
                gc_layer = module
            if type(module) == nn.Conv2d:
                cur_conv = module
        if gc_layer is None:
            print("gradCAM: target layer not found, take the final conv layer")
            gc_layer = cur_conv
        ggc = captum.attr.GuidedGradCam(model, gc_layer)
        method = "GGC"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(ggc)
            method = "GGCwNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM ! n_sample/n_steps 10/10 already gives out of memory error for 8G GPU
            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            attribution = ggc.attribute(transformed_img, target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if deeplift is not None:
        #  ************************************deep lift (Rescale rule)*****************************************
        print("Attribution with deepLift")

        dplift = captum.attr.DeepLift(model)
        method = "deeplift"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(dplift)
            method = "deepliftWNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM
            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx, baselines=None)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            # check documentation or deep lift paper on what the baseline should be, default is zero tensor which will be very similar to IG?
            attribution = dplift.attribute(transformed_img, target=pred_label_idx, baselines=None)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if ig is not None:
        #  ************************************integrated gradient*****************************************
        print("Attribution with integrated gradient")

        integrated_gradients = captum.attr.IntegratedGradients(model)
        method = "IG"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(integrated_gradients)
            method = "IGwNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM ! n_sample/n_steps 10/10 already gives out of memory error for 8G GPU
            attribution = noise_tunnel.attribute(transformed_img, n_samples=5, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx, n_steps=10)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            attribution = integrated_gradients.attribute(transformed_img, target=pred_label_idx, n_steps=50)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if ixg is not None:
        #  ************************************input x gradient*****************************************
        print("Attribution with input x gradient")

        inpXgrad = captum.attr.InputXGradient(model)
        method = "IxG"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(inpXgrad)
            method = "IxGwNoise"
            # the current parameters are placeholders i.e. not enough to generate good result but only visualize gaussian noise!
            # before changing them make sure there is enough RAM
            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

        else:
            attribution = inpXgrad.attribute(transformed_img, target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)

    if deconv is not None:
        #  ************************************deconvolution*****************************************
        print("Attribution with deconvolution")

        dcv = captum.attr.Deconvolution(model)
        method = "deconv"
        if nt is not None:  # noise tunnel version
            noise_tunnel = captum.attr.NoiseTunnel(dcv)
            method = "deconv"

            attribution = noise_tunnel.attribute(transformed_img, n_samples=20, nt_type='smoothgrad_sq',
                                                 target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)
        else:
            attribution = dcv.attribute(transformed_img, target=pred_label_idx)
            save_attr_result(attribution, transformed_img, img_path, method, predicted_label, prediction_score)


if __name__ == "__main__":
    t_start = dt.datetime.now()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="Path to model file")
    ap.add_argument("-i", "--img", required=True,
                    help="folder to test set images")
    ap.add_argument("-tr", "--transformation", required=False, default="MRI_2CV_174x208_pad_299",
                    help="transform config")
    ap.add_argument("-cn", "--className", required=False, default="no,yes",
                    help="class name comma separated")
    ap.add_argument('--occ', action='store_const', const=1, required=False, default=None, help="occlusion sensitivity")
    ap.add_argument('--fab', action='store_const', const=1, required=False, default=None, help="feature ablation")
    ap.add_argument('--fam', required=False, default=None, help="feature map for feature ablation")
    ap.add_argument('--vbp', action='store_const', const=1, required=False, default=None, help="vanilla backprop")
    ap.add_argument('--gcam', action='store_const', const=1, required=False, default=None, help="gradCAM")
    ap.add_argument('--gcl', required=False, default=None, help="target layer for gradcam/guided grad cam")
    ap.add_argument('--gbp', action='store_const', const=1, required=False, default=None, help="guided backprop")
    ap.add_argument('--ggc', action='store_const', const=1, required=False, default=None, help="guided gradCAM")
    ap.add_argument('--deeplift', action='store_const', const=1, required=False, default=None, help="deeplift")
    ap.add_argument('--ixg', action='store_const', const=1, required=False, default=None, help="input x gradient")
    ap.add_argument('--ig', action='store_const', const=1, required=False, default=None, help="integrated gradient")
    ap.add_argument('--deconv', action='store_const', const=1, required=False, default=None, help="deconvolution")
    ap.add_argument('--nt', action='store_const', const=1, required=False, default=None, help="noise tunnel")
    ap.add_argument('--alb', action='store_const', const=1, required=False, default=None, help="albumentation augmentation used")

    # classes = ['female', 'male']  # 0,1
    # model_path = "/home/mw/Analyses/deeplearning_UKB_mri/sex/resnet50_2out_UKBBsubsetDL13_b30_e24_lr2e-4_20052020.pth"
    # img_path = "/home/mw/Analyses/attention_map_test/1325314.1.3.12.2.1107.5.2.18.141243.2018021412535093164416835.dcm.png"
    # transform_setting = "MRI_2CV_174x208_pad_224"

    args = vars(ap.parse_args())
    model_path = args['model']
    image_path = args['img']
    transformation_config = args['transformation']
    cls_names = list(args['className'].split(","))
    # get attribution methods
    occ = args['occ']
    fab = args['fab']
    fam = args['fam']
    vbp = args['vbp']
    gc = args['gcam']
    gcl = args['gcl']
    gbp = args['gbp']
    ggc = args['ggc']
    deeplift = args['deeplift']
    ixg = args['ixg']
    ig = args['ig']
    dcv = args['deconv']
    nt = args['nt']
    albumentation_aug=args["alb"]
    print("Arguments accpeted: {}".format(args))

    # load resnet
    print("Evaluate image:{} \n load model from path:{}".format(image_path, model_path))

    if albumentation_aug is not None:
        from pytorch.transform_config_albumentation import data_transforms
        # print(data_transforms)
    else:
        from pytorch.transform_config import data_transforms



    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(cls_names))
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    # print(model)
    device = get_device()
    model.to(device)
    model.eval()

    # get the attribution maps
    generate_attr_maps(model=model, img_path=image_path, classes=cls_names, transform_config=transformation_config,
                       occlusionSensitivity=occ, featureAblation=fab, featureMask=fam, saliency=vbp, gradCAM=gc,
                       gradCAMlayer=gcl, guidedBackProp=gbp, guidedGradCAM=ggc, deepLift=deeplift, inputXGradient=ixg,
                       integratedGradient=ig, deconvolution=dcv, noiseTunnel=nt)
    delta = dt.datetime.now() - t_start
    print("Attribution map done, run time:{}".format(delta))
