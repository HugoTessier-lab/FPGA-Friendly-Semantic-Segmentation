from models.resnet_unet import resnet18_unet


def load_models(model,
                num_classes,
                pretrained_encoder,
                output_downsampling_rate,
                entry_downsampling_rate):
    if model == 'resnet18_unet':
        return resnet18_unet(
            num_classes=num_classes,
            pretrained_encoder=pretrained_encoder,
            output_downsampling_rate=output_downsampling_rate,
            entry_downsampling_rate=entry_downsampling_rate)
    else:
        print('Invalid model type')
        raise ValueError
