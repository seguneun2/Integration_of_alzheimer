import numpy as np
import torch
import argparse
import os
from torch.nn import functional as F
from torch import nn, optim
from models.classifier import Classifier2
from utils.data import MRIwithGeneDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from config import *


def train_step(context, data):
    classifier = context["classifier"]
    optimizer = context["optimizer"]

    optimizer.zero_grad()

    mri, labels, gene = data["x"], data["y"].float(), data["gene"]
    preds, _, mri_global_feature, gene_local_feature, gene_global_feature = classifier(mri, gene, return_features=True)


    loss_clf = classifier.compute_gene_classification_loss(gene_local_feature, labels) \
              + classifier.compute_classification_loss(preds, labels)
    
    loss_contrast = classifier.compute_contrastive_loss(mri_global_feature, gene_global_feature)
    loss = loss_clf + 0.5*loss_contrast

    loss.backward()
    optimizer.step()

    labels = labels.cpu().numpy()
    preds = preds.detach().cpu().numpy()
    preds[preds < 0.5] = 0
    preds[preds >= 0.5] = 1
    acc = accuracy_score(labels.reshape(-1), preds.reshape(-1))

    return {
        "loss": loss.item(),
        "acc": acc,
    }


@torch.no_grad()
def eval_step(context, data):
    classifier = context["classifier"]

    mri, labels, gene = data["x"], data["y"].float(), data["gene"]
    
    preds =  classifier(mri, gene, return_features=False)
    loss_clf = classifier.compute_classification_loss(preds, labels)

    preds, _, mri_global_feature, gene_local_feature, gene_global_feature = classifier(mri, gene, return_features=True)



    loss_clf = classifier.compute_gene_classification_loss(gene_local_feature, labels) \
              + classifier.compute_classification_loss(preds, labels)
    
    loss_contrast = classifier.compute_contrastive_loss(mri_global_feature, gene_global_feature)
    loss = loss_clf + 0.5*loss_contrast


    labels = labels.cpu().numpy()
    preds = preds.detach().cpu().numpy()
    preds[preds < 0.5] = 0
    preds[preds >= 0.5] = 1
    acc = accuracy_score(labels.reshape(-1), preds.reshape(-1))


    return {
        "loss": loss.item(),
        "acc": acc,

    }


def train(args):
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    trainset = MRIwithGeneDataset("train")
    validset = MRIwithGeneDataset("val")
    

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classifier = Classifier2(in_channels=1, 
                            feature_dim=128, 
                            num_slices=10).to(device)
   
    optimizer = optim.Adam([
        *classifier.parameters(),
    ], lr=args.lr)

    context = {
        "classifier": classifier,
        "optimizer": optimizer
    }

    min_val_loss = np.inf
    max_val_acc = 0.0

    for e in tqdm(range(args.epochs), desc="Epoch", leave=False):
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        classifier.train()

        for x, mask, y, gene in tqdm(train_loader, desc="Train step", leave=False):
            data = {
                "x": x.to(device),
                "y": y.float().to(device),
                "gene": gene.to(device)
            }
            ret = train_step(context, data)
            train_loss += ret["loss"]/len(train_loader)
            train_acc += ret["acc"]/len(train_loader)

        classifier.eval()

        for x, mask, y, gene in tqdm(valid_loader, desc="Evaluation step", leave=False):
            data = {
                "x": x.to(device),
                "y": y.float().to(device),
                "gene": gene.to(device)
            }
            ret = eval_step(context, data)
            valid_loss += ret["loss"]/len(valid_loader)
            valid_acc += ret["acc"]/len(valid_loader)

        print(f"Epochs {e + 1}/{args.epochs}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Train acc: {train_acc:.8f}")
        print(f"Valid loss: {valid_loss:.8f}")
        print(f"Valid acc: {valid_acc:.8f}")

        save_path = os.path.splitext(args.save_path)[0]

        if max_val_acc < valid_acc:
            max_val_acc = valid_acc
            torch.save(context, f"{save_path}_{e}_best_acc.pt")

        if min_val_loss > valid_loss:
            min_val_loss = valid_loss
            torch.save(context, f"{save_path}_best_loss.pt")

        torch.save(context, f"{save_path}_latest.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


