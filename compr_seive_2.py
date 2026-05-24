# compression_sieve.py
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from typing import List, Dict, Tuple
import torchvision.transforms as transforms



def build_mask_from_classes(dataset, class_ids):
    """
    Returns a boolean mask selecting samples whose labels are in class_ids.
    """
    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    mask = torch.tensor([t in class_ids for t in targets])
    return mask

class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, class_map):
        self.dataset = dataset
        self.indices = indices
        self.class_map = class_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.class_map[int(y)]

class RemappedSubset_transform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, class_map, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        # apply transform HERE
        if self.transform is not None:
            x = self.transform(x)

        return x, self.class_map[int(y)]

class ClassSubsetSieve:
    """
    Compression sieve for class-dependent model compression.

    Responsibilities:
      1) Define subsets of similar classes
      2) Build class-restricted dataloaders
      3) Evaluate models on subsets
      4) Support subset-specific fine-tuning
    """

    def __init__(
        self,
        train_dataset,
        test_dataset,
        device="cuda",
        batch_size=64,
        num_workers=2
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_indices_by_class = self._group_indices_by_class(train_dataset)
        self.test_indices_by_class = self._group_indices_by_class(test_dataset)

    # -------------------------------------------------------
    # 1. GROUP DATASET INDICES BY CLASS
    # -------------------------------------------------------
    def _group_indices_by_class(self, dataset):
        class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            class_to_indices[int(y)].append(idx)
        return class_to_indices

    # -------------------------------------------------------
    # 2. DEFINE CLASS SUBSETS (MANUAL OR AUTOMATIC)
    # -------------------------------------------------------
    def define_manual_subsets(self, class_groups: Dict[str, List[int]]):
        """
        Example:
        {
            "vehicles": [1, 9],
            "animals": [2, 3, 4, 5, 6, 7],
            "others": [0, 8]
        }
        """
        return class_groups

    def define_subsets_from_confusion(
        self,
        confusion_matrix: np.ndarray,
        threshold: float = 0.2
    ) -> Dict[str, List[int]]:
        """
        Automatically group classes that are frequently confused.
        """
        num_classes = confusion_matrix.shape[0]
        used = set()
        subsets = {}

        group_id = 0
        for i in range(num_classes):
            if i in used:
                continue

            similar = [i]
            for j in range(num_classes):
                if i != j and confusion_matrix[i, j] > threshold:
                    similar.append(j)

            for c in similar:
                used.add(c)

            subsets[f"group_{group_id}"] = similar
            group_id += 1

        return subsets

    # -------------------------------------------------------
    # 3. BUILD CLASS-CONDITIONAL DATALOADERS
    # -------------------------------------------------------
    # def build_subset_loaders(self, class_ids: List[int]):
    #     train_idx = []
    #     test_idx = []

    #     for c in class_ids:
    #         train_idx.extend(self.train_indices_by_class[c])
    #         test_idx.extend(self.test_indices_by_class[c])

    #     train_subset = Subset(self.train_dataset, train_idx)
    #     test_subset = Subset(self.test_dataset, test_idx)

    #     train_loader = DataLoader(
    #         train_subset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers
    #     )

    #     test_loader = DataLoader(
    #         test_subset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers
    #     )

    #     return train_loader, test_loader

    def build_mask_from_classes(dataset, class_ids):
        """
        Returns a boolean mask selecting samples whose labels are in class_ids.
        """
        targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
        mask = torch.tensor([t in class_ids for t in targets])
        return mask

    # def build_subset_loaders_2(self, class_ids):
    #     mask = build_mask_from_classes(self.train_dataset, class_ids)

    #     subset_indices = torch.nonzero(mask).squeeze().tolist()

    #     subset_trainset = torch.utils.data.Subset(
    #         self.train_dataset, subset_indices
    #     )

    #     subset_test_indices = torch.nonzero(
    #         build_mask_from_classes(self.test_dataset, class_ids)
    #     ).squeeze().tolist()

    #     subset_testset = torch.utils.data.Subset(
    #         self.test_dataset, subset_test_indices
    #     )

    #     trainloader = torch.utils.data.DataLoader(
    #         subset_trainset, batch_size=self.batch_size,
    #         shuffle=True, num_workers=2
    #     )

    #     testloader = torch.utils.data.DataLoader(
    #         subset_testset, batch_size=self.batch_size,
    #         shuffle=False, num_workers=2
    #     )

    #     return trainloader, testloader

    # with remapped classes for models to have number for output classes depending on the subset:
    def build_subset_loaders_2(self, class_ids):
        class_map = {c: i for i, c in enumerate(class_ids)}

        # ---- TRAIN ----
        train_mask = build_mask_from_classes(self.train_dataset, class_ids)
        train_indices = torch.nonzero(train_mask).squeeze().tolist()

        train_subset = RemappedSubset(
            self.train_dataset,
            train_indices,
            class_map
        )

        # ---- TEST ----
        test_mask = build_mask_from_classes(self.test_dataset, class_ids)
        test_indices = torch.nonzero(test_mask).squeeze().tolist()

        test_subset = RemappedSubset(
            self.test_dataset,
            test_indices,
            class_map
        )

        trainloader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        testloader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return trainloader, testloader


    def build_subset_loaders_2_c100(self, class_ids):

        # ---- CIFAR-100 + VGG16 transforms (224×224) ----
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            ),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761)
            ),
        ])

        class_map = {c: i for i, c in enumerate(class_ids)}

        # ---- TRAIN ----
        train_mask = build_mask_from_classes(self.train_dataset, class_ids)
        train_indices = torch.nonzero(train_mask).squeeze().tolist()

        train_subset = RemappedSubset_transform(
            self.train_dataset,
            train_indices,
            class_map,
            transform=transform_train
        )

        # ---- TEST ----
        test_mask = build_mask_from_classes(self.test_dataset, class_ids)
        test_indices = torch.nonzero(test_mask).squeeze().tolist()

        test_subset = RemappedSubset_transform(
            self.test_dataset,
            test_indices,
            class_map,
            transform=transform_test
        )

        trainloader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        testloader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return trainloader, testloader

    # -------------------------------------------------------
    # 4. EVALUATE MODEL ON SUBSET
    # -------------------------------------------------------
    @torch.no_grad()
    def evaluate_on_subset(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

        return 100.0 * correct / max(total, 1)

    # -------------------------------------------------------
    # 5. OPTIONAL FINE-TUNING ON CLASS SUBSET
    # -------------------------------------------------------
    def finetune_on_subset_2(
        self,
        model,
        train_loader,
        epochs=5,
        lr=1e-4,
        weight_decay=0.0
    ):
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        for ep in range(epochs):
            total_loss = 0.0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Subset FT] Epoch {ep+1}/{epochs}, Loss={total_loss:.4f}")

        return model

    def finetune_on_subset(
    self,
    model,
    train_loader,
    test_loader=None,          # subset test loader
    epochs=5,
    lr=1e-4,
    weight_decay=0.0,
    log_interval=1
    ):
        """
        Fine-tune a compressed model on a class subset.

        Prints:
          - training loss
          - training accuracy
          - subset test accuracy (if test_loader provided)
        per epoch.
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        for ep in range(1, epochs + 1):
            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION (SUBSET)
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                if test_acc is not None:
                    print(
                        f"[Subset FT] Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Subset Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"[Subset FT] Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

        return model

    def finetune_on_subset_sgd(
    self,
    model,
    train_loader,
    test_loader=None,          # subset test loader
    epochs=5,
    lr=1e-4,
    weight_decay=0.0,
    log_interval=1
    ):
        """
        Fine-tune a compressed model on a class subset.

        Prints:
          - training loss
          - training accuracy
          - subset test accuracy (if test_loader provided)
        per epoch.
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum = 0.9,
            weight_decay=weight_decay
        )

        for ep in range(1, epochs + 1):
            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION (SUBSET)
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                if test_acc is not None:
                    print(
                        f"[Subset FT] Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Subset Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"[Subset FT] Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

        return model

# jan17th added to imporve vgg c10
    def finetune_sgd(
        self,
        model,
        train_loader,
        test_loader=None,          # CIFAR-10 test loader
        epochs=160,
        lr=0.01,
        weight_decay=5e-4,
        milestones=(80, 120),
        log_interval=1
    ):
        """
        Fine-tune VGG16 on CIFAR-10 using best-practice hyperparameters.

        Uses:
        - SGD + momentum
        - MultiStep LR schedule
        - Strong CIFAR-10 training protocol

        Prints:
        - training loss
        - training accuracy
        - test accuracy (if test_loader provided)
        per epoch.
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(milestones),
            gamma=0.1
        )

        for ep in range(1, epochs + 1):
            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                if test_acc is not None:
                    print(
                        f"[VGG16-C10] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.5f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"[VGG16-C10] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.5f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

            scheduler.step()

        return model

    def finetune_vgg16_cifar100_sgd(
        self,
        model,
        train_loader,
        test_loader=None,          # CIFAR-100 test loader or subset test loader
        epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        milestones=(60, 120, 160),
        log_interval=1
    ):
        """
        Fine-tune VGG16 on CIFAR-100 using best-practice hyperparameters.

        Uses:
        - SGD + momentum + Nesterov
        - MultiStep learning-rate decay
        - CIFAR-100-appropriate schedule

        Prints per epoch:
        - training loss
        - training accuracy
        - test accuracy (if provided)
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(milestones),
            gamma=0.1
        )

        for ep in range(1, epochs + 1):
            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                if test_acc is not None:
                    print(
                        f"[VGG16-C100] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.5f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"[VGG16-C100] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.5f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

            scheduler.step()

        return model


    def finetune_vgg16_cifar10_adam(
    self,
    model,
    train_loader,
    test_loader=None,          # CIFAR-10 test loader or subset test loader
    epochs=30,
    lr=1e-4,
    weight_decay=5e-4,
    log_interval=1
    ):
        """
        Fine-tune VGG16 on CIFAR-10 using Adam optimizer.

        Recommended for:
        - compressed VGG16
        - subset fine-tuning
        - short fine-tuning runs

        Prints per epoch:
        - training loss
        - training accuracy
        - test accuracy (if provided)
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        for ep in range(1, epochs + 1):
            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                if test_acc is not None:
                    print(
                        f"Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"Epoch {ep:03d}/{epochs} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

        return model

    def finetune_sgd_anet_c100(
    self,
    model,
    train_loader,
    test_loader=None,
    epochs=160,
    lr=1e-4,                      # starting lr
    lr_step=10,                   # decay every 10 epochs
    lr_gamma=0.1,                 # decay factor
    weight_decay=5e-4,
    momentum=0.9,
    log_interval=1
    ):
        """
        Fine-tune AlexNet on CIFAR-100.

        Uses:
        - SGD + momentum + Nesterov
        - Step LR decay (every lr_step epochs)
        - Stable CIFAR-100 training protocol
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_step,
            gamma=lr_gamma
        )

        for ep in range(1, epochs + 1):

            # -----------------
            # TRAIN
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVAL
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOG
            # -----------------
            if ep % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                if test_acc is not None:
                    print(
                        f"[AlexNet-C100] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.6f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"[AlexNet-C100] Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.6f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

            scheduler.step()

        return model

    def finetune_vgg16_cifar10_adam_2(
        self,
        model,
        train_loader,
        test_loader=None,
        epochs=30,
        lr=1e-4,
        weight_decay=5e-4,
        lr_step=10,          # decay interval
        lr_gamma=0.1,        # decay factor
        log_interval=1
        ):
        """
        Fine-tune VGG16 on CIFAR-10 using Adam optimizer
        with step-wise LR decay.
        """

        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_step,
            gamma=lr_gamma
        )

        for ep in range(1, epochs + 1):

            # -----------------
            # TRAINING
            # -----------------
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_acc = 100.0 * correct / max(total, 1)
            avg_loss = running_loss / max(len(train_loader), 1)

            # -----------------
            # EVALUATION
            # -----------------
            if test_loader is not None:
                test_acc = self.evaluate_on_subset(model, test_loader)
            else:
                test_acc = None

            # -----------------
            # LOGGING
            # -----------------
            if ep % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                if test_acc is not None:
                    print(
                        f"Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.6f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Test Acc: {test_acc:.2f}%"
                    )
                else:
                    print(
                        f"Epoch {ep:03d}/{epochs} | "
                        f"LR: {lr_now:.6f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}%"
                    )

            scheduler.step()   # 🔹 LR decay step

        return model
