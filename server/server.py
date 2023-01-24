import matplotlib.pyplot as plt
import multiprocessing as mp
import torch, os, copy
import numpy as np
from metrics.miou import StreamSegMetrics
from collections import OrderedDict
from tqdm import tqdm


class Server:
    def __init__(self,
                 server_id: str,
                 model,
                 device,
                 clients=None,
                 local_epochs=None,
                 hypers=None,
                 num_classes=None
                ):
        
        self.model = model
        self.local_epochs = local_epochs
        self.hypers = hypers
        self.device = device
        self.num_classes = num_classes if num_classes is not None else 19
        
        self.save_dir = os.path.join("saved", "autosave", server_id)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        
        self.clients = clients if clients is not None else []
        self.updates = []
        
    def run_fedAvg(self, rounds, T=None, clients_per_round=20):
        print("Running FedAvg training...")
        for t in tqdm(range(rounds), desc="Rounds: "):
            
            if T is not None and T > 0:
                for client in self.clients:
                    if t % T == 0 and client.is_student:
                        client.set_teacher = self.model
            
            client_set = self.select_clients(t, self.clients, clients_per_round)    
            
            for client in client_set:
                self.update_client(client, t)
            
            self.update_model()            
            
            checkpoint = {
                    "round": t,
                    "model_state_dict": self.model.state_dict(),
                }
            
            self.save_model(checkpoint=checkpoint)
    
    def update_client(self, client, cur_round):
        client.model.load_state_dict(self.model_params_dict)
        num_samples, update = client.train(cur_round=cur_round, prog_bar=False)
        self.add_update(num_samples, update)

    
    def add_update(self, num_samples, update):
        self.updates.append((num_samples, update))
    
    def update_model(self):

        averaged_state_dict = self._aggregation()
        self.model.load_state_dict(averaged_state_dict, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []
    
    def _aggregation(self):
        
        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to(self.device) / total_weight

        return averaged_sol_n
    
    def put_client(self, client):
        self.clients.append(client)
        print("Client list updated.")
        return 
    
    def select_clients(self, my_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        #np.random.seed(my_round)
        return np.random.choice(possible_clients, num_clients, replace=False)
        
    def save_model(self, save_path=None, checkpoint=None):
        '''
        Use save_model(path) to save the state_dict in the path. Pass only the checkpoint to save the checkpoint in autosave/serverID/.
        Other uses are not implemented yet.
        '''
        if save_path is not None and checkpoint is None:
            torch.save(self.model.state_dict(), save_path)
        
        elif save_path is None and checkpoint is not None:
            save_dir = self.save_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, "round" + str(checkpoint["round"]) + ".pth")
            torch.save(checkpoint, save_path)
        
        else:
            raise NotImplementedError("Not implemented yet. Take a look to the source code to know how to use save_model()")
    
    def load_model(self, load_path):

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_params_dict = copy.deepcopy(self.model.state_dict())
            
            self.model.eval()
            print(f"Loaded model: {load_path}")
            return True
    
        else:
            print(f"Model {load_path} not found")
            return False
        
    
    def test(self, dataloader, cl:str=None):
        
        self.model.eval()
        images, labels = next(iter(dataloader))
        images = images.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device, dtype=torch.long)
        
        if self.output_aux:
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
