from utils.train import train_model
from utils.optimizer import load_optimizer
from utils.scheduler import load_scheduler
from utils.criterion import load_criterion
from utils.checkpoint import Checkpoint
from utils.metrics import load_metrics
from datasets.load_dataset import load_dataset
from models.load_models import load_models
import argparse


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def parse_arguments():
    parser = argparse.ArgumentParser(description='SWD')

    # General

    parser.add_argument("--device", default='cuda',
                        help="Device on which to run training (default: 'cuda')")

    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Distributes the model across available GPUs.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode.")

    parser.add_argument('--output_path', type=str, default="./outputs",
                        help="Where to save models  (default: './checkpoint')")

    parser.add_argument('--name', type=str, default="resnet18_unet_training",
                        help="Name of the experiment  (default: 'resnet18_unet_training')")

    # Dataset

    parser.add_argument('--dataset', type=str, default="cityscapes",
                        help="Which dataset to choose (default: 'cityscapes')")

    parser.add_argument('--dataset_root', type=str, default="",
                        help="Path to the dataset (default: '')")

    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='Input batch size for training (default: 1)')

    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Input batch size for testing (default: 1)')

    parser.add_argument('--train_crop_size', type=tuple_type, default=(512, 1024),
                        help="Cropped image size for training (default: (512, 1024))")

    parser.add_argument('--test_crop_size', type=tuple_type, default=(1024, 2048),
                        help="Cropped image size for testing (default: (1024, 2048))")

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers to load the dataset (default: 4)')

    parser.add_argument('--train_label_downsample_rate', type=int, default=1,
                        help='By which factor to decrease the resolution of training labels, after cropping (default: 1)')

    parser.add_argument('--train_image_downsample_rate', type=int, default=1,
                        help='By which factor to decrease the resolution of training images, after cropping (default: 1)')

    parser.add_argument('--test_label_downsample_rate', type=int, default=1,
                        help='By which factor to decrease the resolution of testing labels, after cropping (default: 1)')

    parser.add_argument('--test_image_downsample_rate', type=int, default=1,
                        help='By which factor to decrease the resolution of testing images, after cropping (default: 1)')

    # Model

    parser.add_argument('--model', type=str, default="resnet18_unet",
                        help="Which model to choose  (default: 'resnet18_unet')")

    parser.add_argument("--pretrained_encoder", action="store_true", default=False,
                        help="Whether to load imagenet weights for the encoder or not.")

    parser.add_argument('--output_downsampling_rate', type=int, default=1,
                        help='Rate by which the output of a network is smaller than its input (default: 1)')

    parser.add_argument('--entry_downsampling_rate', type=int, default=4,
                        help='Rate by which the first layers of a network downsample the input (default: 4)')

    # Training

    parser.add_argument('--optimizer', type=str, default="sgd",
                        help="Which optimizer to choose  (default: 'sgd')")

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')

    parser.add_argument('--wd', default="5e-4", type=float,
                        help='Weight decay rate (default: 5e-4)')

    parser.add_argument('--scheduler', type=str, default="poly",
                        help="Type of scheduler (default: 'poly')")

    parser.add_argument('--poly_exp', type=float, default=2,
                        help='Polynomial exponent of scheduler (default: 2)')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')

    parser.add_argument('--criterion', type=str, default="rmi",
                        help="Type of criterion (default: 'rmi')")

    parser.add_argument('--metrics', type=str, nargs='+', default=['miou'],
                        help="List of metrics (default: ['miou'])")

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    dataset, num_classes = load_dataset(
        dataset=arguments.dataset,
        train_batch_size=arguments.train_batch_size,
        test_batch_size=arguments.test_batch_size,
        train_crop_size=arguments.train_crop_size,
        test_crop_size=arguments.test_crop_size,
        dataset_root=arguments.dataset_root,
        workers=arguments.workers,
        train_label_downsample_rate=arguments.train_label_downsample_rate,
        train_image_downsample_rate=arguments.train_image_downsample_rate,
        test_label_downsample_rate=arguments.test_label_downsample_rate,
        test_image_downsample_rate=arguments.test_image_downsample_rate)

    model = load_models(
        model=arguments.model,
        num_classes=num_classes,
        pretrained_encoder=arguments.pretrained_encoder,
        output_downsampling_rate=arguments.output_downsampling_rate,
        entry_downsampling_rate=arguments.entry_downsampling_rate)
    optimizer = load_optimizer(
        optimizer=arguments.optimizer,
        model=model,
        learning_rate=arguments.lr,
        weight_decay=arguments.wd)
    scheduler = load_scheduler(
        optimizer=optimizer,
        scheduler=arguments.scheduler,
        epochs=arguments.epochs,
        poly_exp=arguments.poly_exp)
    checkpoint = Checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=arguments.device,
        distributed=arguments.distributed,
        save_folder=arguments.output_path,
        name=arguments.name)

    criterion = load_criterion(criterion=arguments.criterion)
    metrics = load_metrics(metrics=arguments.metrics)

    train_model(
        checkpoint=checkpoint,
        dataset=dataset,
        epochs=arguments.epochs,
        output_path=arguments.output_path,
        debug=arguments.debug,
        criterion=criterion,
        device=arguments.device,
        name='Training',
        metrics=metrics)


if __name__ == '__main__':
    main()
