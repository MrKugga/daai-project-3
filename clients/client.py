from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch, os, copy, wandb
from metrics.miou import StreamSegMetrics
from tqdm import tqdm

LOG_FREQUENCY = 50

class Client:
    def __init__(self,
                 client_id: str,
                 dataset,
                 model,
                 batch_size,
                 device,
                 epochs,
                 hypers,
                 num_classes=None,
                 autosave=True,
                 wb_log=True
                ):
        
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.hypers = hypers 
        self.num_classes = num_classes if num_classes is not None else 19
        self.save_dir = os.path.join("saved", "autosave", client_id)
        self.autosave = autosave
        self.wb_log = wb_log
        
        
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, drop_last=True)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.SGD(self.model.parameters(), lr=hypers["LR"], momentum=hypers["MOMENTUM"], weight_decay=hypers["WEIGHT_DECAY"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=hypers["STEP_SIZE"], gamma=hypers["GAMMA"])
        
    def calc_losses(self, images, labels):
        
        if self.model.output_aux:
            
            outputs, feat2, feat3, feat4, feat5_4 = self.model(images)
            loss = self.criterion(outputs, labels)
            boost_loss = 0
            boost_loss += self.criterion(feat2, labels)
            boost_loss += self.criterion(feat3, labels)
            boost_loss += self.criterion(feat4, labels)
            boost_loss += self.criterion(feat5_4, labels)

            loss_tot = loss + boost_loss
            dict_calc_losses = {'loss': loss, 'boost_loss': boost_loss, 'loss_tot': loss_tot}
            
        else:
            outputs = self.model(images)
            loss_tot = self.criterion(outputs, labels)
            dict_calc_losses = {'loss_tot': loss_tot}
            
        return dict_calc_losses, outputs
    
    def _run_epoch(self, cur_epoch, optimizer, scheduler=None, cur_round=None):  
        for cur_step, (images, labels) in enumerate(self.data_loader):
            
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            dict_calc_losses, outputs = self.calc_losses(images, labels)
            
            # mIoU accuracy
            accuracy = self.update_metrics(outputs, labels, self.num_classes)
            dict_calc_losses['loss_tot'].backward()
            optimizer.step()
            
            # Logging
            if self.wb_log:
                wandb.log({'accuracy': accuracy["Mean IoU"], 
                           'loss': dict_calc_losses["loss_tot"], 
                           'epoch': cur_epoch,
                           'round': cur_round if cur_round is not None else -1
                          })
        
        if scheduler is not None:
            scheduler.step()
        
        return dict_calc_losses, accuracy
    
    def save_model(self, save_path=None, checkpoint=None):
        '''
        Use save_model(path) to save the state_dict in the path. Pass only the checkpoint to save the checkpoint in autosave/clientID/.
        Other uses are not implemented yet.
        '''
        if save_path is not None and checkpoint is None:
            torch.save(self.model.state_dict(), save_path)
        
        elif save_path is None and checkpoint is not None:
            save_dir = self.save_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, "epoch" + str(checkpoint["epoch"]) + ".pth")
            torch.save(checkpoint, save_path)
        
        else:
            raise NotImplementedError("Not implemented yet. Take a look to the source code to know how to use save_model()")
        
        
        return save_path
    
    # Da implementare load completo non solo state dict ma anche optimizer state dict
    def load_model(self, load_path, state_dict=None):
        
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                self.model.eval()
                return epoch
            except KeyError:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                return None
                
        else:
            print(f"Model {load_path} not found")
    
    def train(self, epochs=None, hypers=None, cur_round=None, prog_bar=True):
        epochs = epochs if epochs is not None else self.epochs
        hypers = hypers if hypers is not None else self.hypers
        
        if hypers is not None:
            optimizer = optim.SGD(self.model.parameters(), lr=hypers["LR"], momentum=hypers["MOMENTUM"], weight_decay=hypers["WEIGHT_DECAY"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hypers["STEP_SIZE"], gamma=hypers["GAMMA"])
        else:
            optimizer = self.optimizer
            scheduler = self.scheduler
     
        self.model.train()
        
        if prog_bar:
            print(f"ID: {self.client_id} - Training...")
            ep_iterable = tqdm(range(epochs), desc="Epoch: ")
        else:
            ep_iterable = range(epochs)
            
        for epoch in ep_iterable:
            dict_calc_losses, accuracy = self._run_epoch(epoch, optimizer, scheduler, cur_round)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            if self.autosave: self.save_model(checkpoint=checkpoint)

        update = copy.deepcopy(self.model.state_dict())
        num_samples = len(self.dataset)

        return num_samples, update
    
    def test(self, dataloader=None, cl:str=None):
        self.model.eval()
        
        dataloader = dataloader if dataloader is not None else self.data_loader
        
        images, labels = next(iter(dataloader))
        images = images.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)
        
        if self.model.output_aux:
            outputs, feat2, feat3, feat4, feat5_4 = self.model(images)
            
        else:
            outputs = self.model(images)
        
        _, prediction = outputs.max(dim=1)
        
        self.plot_sample(prediction[0], images[0], labels[0], cl)
        accuracy = self.update_metrics(outputs, labels, self.num_classes)
        
        return accuracy
        

    def update_metrics(self, outputs, labels, num_classes):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric = StreamSegMetrics(num_classes)
        metric.update(labels, prediction)
        accuracy = metric.get_results()
        metric.reset()
        
        return accuracy
    

    def plot_sample(self, prediction, image, label, cl: str=None):
        
        image = image.cpu()
        label = label.cpu()
        prediction = prediction.cpu()
        
        found = True
        if cl is not None:
            map_classes = self.dataset.map_classes
            if cl in map_classes.values():
                for cl, name in enumerate(map_classes.values()):
                    if name == cl:
                        mapping_pred = prediction==cl
                        mapping_label = label==cl
            else:
                print("Class not found. Plotting all classes instead.")
                flag = True
            
        elif cl is None or not found:
            mapping_pred = prediction!=255
            mapping_label = label!=255
            
        plt.imshow(image.permute(1,2,0))
        plt.show()
            
        plt.imshow(prediction*mapping_pred+1, cmap="gray")
        plt.show()
            
        plt.imshow(label*mapping_label+1, cmap="gray")
        plt.show()
        

        
        