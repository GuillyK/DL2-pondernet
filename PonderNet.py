from tkinter.filedialog import test
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, USPS
from torchvision import transforms
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from math import floor
import datetime
import argparse

"""

This is the main code contribution of our research into the performance of PonderNeton simple architectures and datasets. 
Our implementation has sourced several implementation aspects from "Pondernet explained" TWD post,
which can be found at "https://towardsdatascience.com/pondernet-explained-5e9571e657d". These aspects include the PonderNet
loss calculation, as well as general functionality such as the datamodule and the model object. These aspects were implemented
in our code to serves our purpose of examining PonderNet's performance.

The main idea is to apply a PonderNet loss to a small CNN architecture, and
monitor its performance on datasets of varying difficulty, and determine wether
PonderNet 

Important objects in this file are:
- PonderDataModule: Datamodule from which all data operations are performed.
- PonderLoss: Loss module to calculate and apply the PonderLoss on any architecture.
- PonderModel: Model module containing the initialization, training and validation
                of the PonderNet Model.

Hyperparameters such as learning rate, epoch count, batch size etc. can be edited above the main loop.
Dataset parameters can be set using the command line:
--dataset: str, select dataset. default is MNIST, select from "MNIST", "FMNIST" and "USPS"
--rotation: bool flag, use rotations. Default is true.
--no-rotation: bool flag, don't use rotations. Default is false.
--ponder: bool flag, use ponderloss. Default is true.
--no-ponder: bool flag, don't use ponderloss. Default is false.


06-06-2022, Joris Hijstek, Guilly Kolkman, Alex Labro.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PonderDataModule(pl.LightningDataModule):
    """
        DataModule to hold the various datasets.
        Parameters
        ----------
        data_dir : str
            Directory where MNIST will be downloaded or taken from.
        train_transform : [transform] 
            List of transformations for the training dataset. The same
            transformations are also applied to the validation dataset.
        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.
        batch_size : int
            Batch size for both all dataloaders.
    """

    def __init__(self, dataset_name="MNIST", data_dir='./', train_transform=None, test_transform=None, batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dataset_name = dataset_name
        self.dataset = {}
        self.num_channels = 1

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28,28)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        """Creates and loads MNIST/FMNIS/USPS/CIFAR10 dataset objects."""
        # download data (train/val and test sets)
        if self.dataset_name == "FMNIST":
            FashionMNIST(self.data_dir, train=True, download=True)
            FashionMNIST(self.data_dir, train=False, download=True)
            self.dataset['dataset'] = FashionMNIST
            self.num_channels = 1
        elif self.dataset_name == "MNIST":
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)
            self.dataset['dataset']  = MNIST
            self.num_channels = 1
        elif self.dataset_name == "CIFAR10":
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)
            self.dataset['dataset']  = CIFAR10
            self.num_channels = 3
            print("set channels to ", self.num_channels)
        elif self.dataset_name == "USPS":
            USPS(self.data_dir, train=True, download=True)
            USPS(self.data_dir, train=False, download=True)
            self.dataset['dataset']  = USPS
            self.num_channels = 1

    def setup(self, stage=None, split_ratio=0.083):
        """
            Assigns datasets per possible stage ("fit","validate", "test").
            Parameters:
            ----------
            split_ratio : float
                Ratio between training and validation split. 
        """
        dataset = self.dataset['dataset']
        # we set up only relevant datasets when stage is specified
        if stage in [None, 'fit', 'validate']:
            mnist_train = dataset(self.data_dir, train=True,
                                transform=(self.train_transform or self.default_transform))
            print("dataset len:",len(mnist_train))
            val_len = int(len(mnist_train) * split_ratio)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [len(mnist_train) - val_len, val_len])
        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.mnist_test = dataset(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.mnist_test = [dataset(self.data_dir,
                                         train=False,
                                         transform=test_transform)
                                   for test_transform in self.test_transform]

    def train_dataloader(self):
        """Returns training dataloader"""
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return mnist_train

    def val_dataloader(self):
        """Returns validation dataloader"""
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2)
        return mnist_val

    def test_dataloader(self):
        """Returns test dataloader"""
        if isinstance(self.mnist_test, FashionMNIST):
            return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=2)

        mnist_test = [DataLoader(test_dataset,
                                 batch_size=self.batch_size)
                      for test_dataset in self.mnist_test]
        return mnist_test

class ReconstructionLoss(nn.Module):
    """
        Computes the weighted average of the given loss across steps according to
        the probability of stopping at each step.
        Parameters
        ----------
        loss_func : callable
            Loss function accepting true and predicted labels. It should output
            a loss item for each element in the input batch.  
    """

    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        """
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, of shape `(max_steps, batch_size)`.
            y_pred : torch.Tensor
                Predicted outputs, of shape `(max_steps, batch_size)`.
            y : torch.Tensor
                True targets, of shape `(batch_size)`.
            Returns
            -------
            total_loss : torch.Tensor
                Scalar representing the reconstruction loss.
        """
        total_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_pred[n], y)).mean()
            total_loss = total_loss + loss

        return total_loss


class RegularizationLoss(nn.Module):
    """
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.
        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.
        max_steps : int
            Maximum number of allowed pondering steps.
    """

    def __init__(self, lambda_p: float, max_steps: int = 1_000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor):
        """
            Compute the loss.
            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.
            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        """
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)


class PonderLoss:
    """
        Class to group the losses together and calculate the total loss.
        Parameters
        ----------
        rec_loss : torch.Tensor
            Reconstruction loss obtained from running the network.
        reg_loss : torch.Tensor
            Regularization loss obtained from running the network.
        beta : float
            Hyperparameter to calculate the total loss.
    """

    def __init__(self, rec_loss, reg_loss, beta):
        self.rec_loss = rec_loss
        self.reg_loss = reg_loss
        self.beta = beta

    def get_rec_loss(self):
        """returns the reconstruciton loss"""
        return self.rec_loss

    def get_reg_loss(self):
        """returns the regularization loss"""
        return self.reg_loss

    def get_total_loss(self):
        """returns the total loss"""
        return self.rec_loss + self.beta * self.reg_loss


class CNN(nn.Module):
    """
        Simple convolutional neural network.
        Parameters
        ----------
        n_input : int
            Size of the input image. We assume the image is a square,
            and `n_input` is the size of one side.
        n_ouptut : int
            Size of the output.
        kernel_size : int
            Size of the kernel.
        num_channels: int
            Size of the input channel of the network.
    """

    def __init__(self, n_input=28, n_output=50, kernel_size=5, num_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # calculate size of convolution output
        self.lin_size = floor((floor((n_input - (kernel_size - 1)) / 2) - (kernel_size - 1)) / 2)
        self.fc1 = nn.Linear(self.lin_size ** 2 * 20, n_output)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        return x


class MLP(nn.Module):
    """
        Simple 3-layer multi layer perceptron.
        Parameters
        ----------
        n_input : int
            Size of the input.
        n_hidden : int
            Number of units of the hidden layer.
        n_ouptut : int
            Size of the output.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        """forward pass"""
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x

class PonderModel(pl.LightningModule):
    """
        PonderNet variant to perform image classification on MNIST. It is capable of
        adaptively choosing the number of steps for which to process an input.
        Parameters
        ----------
        n_hidden : int
            Hidden layer size of the propagated hidden state.
        n_hidden_lin :
            Hidden layer size of the underlying MLP.
        n_hidden_cnn : int
            Hidden layer size of the output of the underlying CNN.
        kernel_size : int
            Size of the kernel of the underlying CNN.
        max_steps : int
            Maximum number of steps the network is allowed to "ponder" for.
        lambda_p : float 
            Parameter of the geometric prior. Must be between 0 and 1.
        beta : float
            Hyperparameter to calculate the total loss.
        lr : float
            Learning rate.
        num_channels: int
            Number of color channels in the used dataset samples.
        ponder: bool
            Boolean that decides if the pondernet method is used or not.
        Modules
        -------
        cnn : CNN
            Learnable convolutional neural network to embed the image into a vector.
        mlp : MLP
            Learnable 3-layer machine learning perceptron to combine the hidden state with
            the image embedding.
        ouptut_layer : nn.Linear
            Linear module that serves as a multi-class classifier.
        lambda_layer : nn.Linear
            Linear module that generates the halting probability at each step.
    """

    def __init__(self, n_hidden, n_hidden_lin, n_hidden_cnn, kernel_size, max_steps,
                lambda_p, beta, lr, num_channels, ponder):
        super().__init__()

        # attributes
        self.n_classes = 10
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.beta = beta
        self.n_hidden = n_hidden
        self.lr = lr
        self.ponder = ponder

        # modules
        self.cnn = CNN(n_input=28, kernel_size=kernel_size, n_output=n_hidden_cnn, num_channels=num_channels)
        self.mlp = MLP(n_input=n_hidden_cnn + n_hidden, n_hidden=n_hidden_lin, n_output=n_hidden)
        self.outpt_layer = nn.Linear(n_hidden, self.n_classes)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        # losses
        self.loss_rec = ReconstructionLoss(nn.CrossEntropyLoss())
        self.loss_reg = RegularizationLoss(self.lambda_p, max_steps=self.max_steps, device=self.device)

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hparams on W&B
        self.save_hyperparameters()

    def forward(self, x):
        """
            Run the forward pass.
            Parameters
            ----------
            x : torch.Tensor
                Batch of input features of shape `(batch_size, n_elems)`.
            Returns
            -------
            y : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the predictions for each step and each sample. In case
                `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.
            p : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the halting probabilities. Sums over rows (fixing a sample)
                are 1. In case `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.
            halting_step : torch.Tensor
                An integer for each sample in the batch that corresponds to
                the step when it was halted. The shape is `(batch_size,)`. The
                minimal value is 1 because we always run at least one step.
        """

        # extract batch size for QoL
        batch_size = x.shape[0]

        # propagate to get h_1
        h = x.new_zeros((batch_size, self.n_hidden))
        embedding = self.cnn(x)
        concat = torch.cat([embedding, h], 1)
        h = self.mlp(concat)

        # lists to save p_n, y_n
        p = []
        y = []

        # vectors to save intermediate values
        un_halted_prob = h.new_ones((batch_size,))  # unhalted probability till step n
        halting_step = h.new_zeros((batch_size,), dtype=torch.long)  # stopping step
        
        # main loop
        for n in range(1, self.max_steps + 1):
            # obtain lambda_n
            if n == self.max_steps:
                lambda_n = h.new_ones(batch_size)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze()

            # obtain output and p_n
            y_n = self.outpt_layer(h)
            p_n = un_halted_prob * lambda_n

            # append p_n, y_n
            p.append(p_n)
            y.append(y_n)

            # calculate halting step
            halting_step = torch.maximum(
                n
                * (halting_step == 0)
                * torch.bernoulli(lambda_n).to(torch.long),
                halting_step)

            # track unhalted probability and flip coin to halt
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # propagate to obtain h_n
            embedding = self.cnn(x)
            concat = torch.cat([embedding, h], 1)
            h = self.mlp(concat)

            # break when pondering if we are in inference and all elements have halting_step
            if self.ponder:
                if not self.training and (halting_step > 0).sum() == batch_size:
                    break

        return torch.stack(y), torch.stack(p), halting_step

    def     _get_loss_and_metrics(self, batch):
        """
            Returns the losses, the predictions, the accuracy and the number of steps.
            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Batch to process.
            Returns
            -------
            loss : Loss
                Loss object from which all three losses can be retrieved.
            preds : torch.Tensor
                Predictions for the current batch.
            acc : torch.Tensor
                Accuracy obtained with the current batch.
            steps : torch.Tensor
                Average number of steps in the current batch.
        """
        # extract the batch
        data, target = batch

        # forward pass
        y, p, halted_step = self(data)

        # remove elements with infinities (after taking the log)
        if torch.any(p == 0) and self.training and self.ponder:
            valid_indices = torch.all(p != 0, dim=0)
            p = p[:, valid_indices]
            y = y[:, valid_indices]
            halted_step = halted_step[valid_indices]
            target = target[valid_indices]


        # Calculate the loss
        loss_rec_ = self.loss_rec(p, y, target)
        loss_reg_ = self.loss_reg(p)
        halted_index = (halted_step - 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, self.n_classes)

        # calculate the predictions
        logits = y.gather(dim=0, index=halted_index).squeeze()
        preds = torch.argmax(logits, dim=1)

        # Calculate loss:
        if self.ponder:
            loss = PonderLoss(loss_rec_, loss_reg_, self.beta)
        else:
            loss = self.loss_rec.forward(p, y, target)

        # Calculate accuracy
        acc = self.accuracy(preds, target)

        # Calculate the average number of steps
        steps = (halted_step * 1.0).mean()

        return loss, preds, acc, steps


    def training_step(self, batch, batch_idx):
        """
            Perform the training step.
            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current training batch to train on.
            Returns
            -------
            loss : torch.Tensor
                Loss value of the current batch.
        """
        loss, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('train/steps', steps)
        self.log('train/accuracy', acc)
        if self.ponder:
            self.log('train/total_loss', loss.get_total_loss())
            self.log('train/reconstruction_loss', loss.get_rec_loss())
            self.log('train/regularization_loss', loss.get_reg_loss())
            return loss.get_total_loss()
        else:
            self.log('train/total_loss', loss)
            return loss

    def validation_step(self, batch, batch_idx):
        """
            Perform the validation step. Logs relevant metrics and returns
            the predictions to be used in a custom callback.
            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current validation batch to evaluate.
            Returns
            -------
            preds : torch.Tensor
                Predictions for the current batch.
        """
        loss, _, acc, steps = self._get_loss_and_metrics(batch)
        # print("validation acc:", acc.item())

        self.log('val/steps', steps)
        self.log('val/accuracy', acc)
        if self.ponder:
            self.log('val/total_loss', loss.get_total_loss())
            self.log('val/reconstruction_loss', loss.get_rec_loss())
            self.log('val/regularization_loss', loss.get_reg_loss())
            return loss.get_total_loss()
        else:
            self.log('val/total_loss', loss)
            return loss

    def test_step(self, batch, batch_idx, dataset_idx=0):
        """
            Perform the test step. Returns relevant metrics.
            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current test batch to evaluate.
            Returns
            -------
            acc : torch.Tensor
                Accuracy for the current batch.
            steps : torch.Tensor
                Average number of steps for the current batch.
        """
        _, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log(f'test_{dataset_idx}/steps', steps)
        self.log(f'test_{dataset_idx}/accuracy', acc)

    def configure_optimizers(self):
        """
            Configure the optimizers and learning rate schedulers.
            Returns
            -------
            config : dict
                Dictionary with `optimizer` and `lr_scheduler` keys, with an
                optimizer and a learning scheduler respectively.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', verbose=True),
                "monitor": 'val/accuracy',
                "interval": 'epoch',
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        """ Returns a list of callbacks """
        model_checkpoint = ModelCheckpoint(monitor="val/accuracy", mode='max')
        return [model_checkpoint]

def get_accuracy(logit, target, batch_size):
        """ Obtain accuracy for training round """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()

def get_transforms():
    # define transformations
    transform_22 = transforms.Compose([
        transforms.RandomRotation(degrees=22.5),
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_45 = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_67 = transforms.Compose([
        transforms.RandomRotation(degrees=67.5),
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_90 = transforms.Compose([
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_transform = transform_22
    test_transform = [transform_22, transform_45, transform_67, transform_90]

    return train_transform, test_transform

# OPTIMIZER SETTINGS
LR = 0.001
GRAD_NORM_CLIP = 0.5
    
# MODEL HPARAMS
N_HIDDEN = 64
N_HIDDEN_CNN = 64
N_HIDDEN_LIN = 64
KERNEL_SIZE = 5
BATCH_SIZE = 64
EPOCHS = 40
MAX_STEPS = 20
LAMBDA_P = 0.2
BETA = 0.01


if __name__ == "__main__":
    # set seeds
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Choose from MNIST, FMNIST, USPS, (CIFAR10)")
    parser.set_defaults(dataset="MNIST")
    parser.add_argument("--rotation", action='store_true', help="Train with rotations or not; boolean")
    parser.add_argument("--no-rotation", dest="rotation", action="store_false", help="Train with rotations or not; boolean")
    parser.set_defaults(rotation=False)
    parser.add_argument('--ponder', action='store_true')
    parser.add_argument('--no-ponder', dest='ponder', action='store_false')
    parser.set_defaults(ponder=True)
    args = parser.parse_args()
    print(args.ponder)

    if args.rotation:
        train_transform, test_transform = get_transforms()
    else:
        train_transform, test_transform = None, None

    # initialize datamodule and model
    dataset = PonderDataModule(batch_size=BATCH_SIZE,
                            train_transform=train_transform,
                            test_transform=test_transform,
                            dataset_name=args.dataset)

    if args.dataset == "CIFAR10":
        num_channels = 3
    else:
        num_channels = 1

    model = PonderModel(n_hidden=N_HIDDEN,
                        n_hidden_cnn=N_HIDDEN_CNN,
                        n_hidden_lin=N_HIDDEN_LIN,
                        kernel_size=KERNEL_SIZE,
                        max_steps=MAX_STEPS,
                        lambda_p=LAMBDA_P,
                        beta=BETA,
                        lr=LR,
                        num_channels=num_channels,
                        ponder=args.ponder)

    # setup logger'
    exp_name = "{}_{}_{}".format(args.dataset, ("Rotation" if args.rotation else "No-rotation"), ("Ponder" if args.ponder else "no-ponder"))
    logger = WandbLogger(project='PonderNet_{}'.format(args.dataset), name=exp_name)
    logger.watch(model)

    trainer = Trainer(
        logger=logger,                      # W&B integration
        gpus=-1,                            # use all available GPU's
        max_epochs=EPOCHS,                  # maximum number of epochs
        gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
        val_check_interval=0.25,            # validate 4 times per epoch
        precision=16,                       # train in half precision
        deterministic=True)                 # for reproducibility

    # fit the model
    trainer.fit(model, datamodule=dataset)

    # evaluate on the test set
    trainer.test(model, datamodule=dataset)