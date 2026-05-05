
import os
import sys
import six

import torchvision.transforms as transforms

def compose_transforms(meta, resize=256, center_crop=True,
                       override_meta_imsize=False):
    
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)

def load_module_2or3(model_name, model_def_path):
    
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod
