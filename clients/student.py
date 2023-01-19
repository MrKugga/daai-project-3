from clients.client import Client
import os, torch, wandb

class StudentClient(Client):
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
                 teacher=None
                ):
        super().__init__(client_id, dataset, model, batch_size, device, epochs, hypers, num_classes, autosave)
        
        self.teacher = teacher
        self.is_student = True
        
    def set_teacher(self, model):
        self.teacher = model
        
    def get_batch_mask(self, outputs):
        
        # Manual normalization of the predictions to have 0-1 probabilities
        prob, pred = outputs.max(dim=1)
        prob -= prob.min(dim=1, keepdim=True)[0]
        prob /= prob.max(dim=1, keepdim=True)[0]
        
        b, _, _ = pred.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(prob, pred)], dim=0)
        return mask
    
    def get_image_mask(self, prob, pl, th=None):
        '''
        If threshold is not specified, it is set to 90%. This assumes that the teacher model is very well pre-trained. 
        Every pixel which is under the threshold is labeled as 255, which is the index that will be ignored when calculating
        losses.
        '''
        th = th if th is not None else 0.9
        exclude = prob < th
        pl[exclude==True] = 255
        
        return pl
        
        
    
    def get_pseudo_labels(self, batch, th=None):
       
        with torch.no_grad():
            if self.teacher.output_aux:
                outputs = self.teacher(batch)[0]
            else:
                outputs = self.teacher(batch)
        
        pl = self.get_batch_mask(outputs)

        return pl
    
    def load_teacher(self, load_path):
        
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.teacher.load_state_dict(checkpoint['model_state_dict'])
            self.teacher.eval()
            return True
    
        else:
            return False  
        
    def _run_epoch(self, cur_epoch, optimizer, scheduler=None, cur_round=None):  
        for cur_step, (images, real_labels) in enumerate(self.data_loader):
            
            # Getting psudo labels from teacher model
            images = images.to(self.device, dtype=torch.float32)
            
            labels = self.get_pseudo_labels(images)
            labels = labels.to(self.device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            dict_calc_losses, outputs = self.calc_losses(images, labels)
            
            # mIoU accuracy
            accuracy = self.update_metrics(outputs, real_labels, self.num_classes)
            accuracy_pseudo = self.update_metrics(outputs, labels, self.num_classes)
            dict_calc_losses['loss_tot'].backward()
            optimizer.step()
            
            # Logging
            if self.wb_log:
                wandb.log({'accuracy': accuracy["Mean IoU"], 
                           'accuracy_pseudo': accuracy_pseudo["Mean IoU"], 
                           'loss': dict_calc_losses["loss_tot"], 
                           'epoch': cur_epoch,
                           'round': cur_round if cur_round is not None else -1
                          })
        
        if scheduler is not None:
            scheduler.step()
        
        return dict_calc_losses, accuracy
        