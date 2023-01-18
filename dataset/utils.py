import os, random, math
import numpy as np

MAX_IMAGES_PER_CLIENT = 20
    

def txt2list(text: str):
  
    lst = []
    for path in open(f"{os.getcwd()}/{text}"):
        path = path.replace("\n","")
        lst.append(path)
    
    # Removing city from path
    for idx, path in enumerate(lst):
        lst[idx] = path.split('/')[-1]

    return lst

def list2txt(lst, path):
    
    with open(path, 'w') as f:
        for image in lst:
            f.write(image)
            f.write('\n')

def random_split(root: str):
    '''
    Root: root of the images
    
    Split for partition A of Step 1.
    '''
    file_names = [x for x in os.listdir(root) if os.path.isfile(os.path.join(root, x))]
    cities = []

    for path in file_names:
        path = path.replace("\n","")
        city = path.split("_")[0]

        if not any(city in sublist for sublist in cities):
            cities.append([city, [path]])
        else:
            for sublist in cities:
                if city == sublist[0]:
                    sublist[1].append(path)
                else:
                    continue

    test = []
    train = []

    for idx, sublist in enumerate(cities):
        img1 = sublist[1].pop(random.randrange(len(sublist[1])))
        img2 = sublist[1].pop(random.randrange(len(sublist[1])))

        test.append(img1)
        test.append(img2)

        for path in sublist[1]:
            train.append(path)
    
    return test, train

def uniform_split(dataset, num_clients):
    '''
    Uniform split described in Step 3.
    Input:
        dataset: Cityscapes, the dataset to divide between clients
        num_clients: int, number of train clients to which the dataset will be divided
        
    Output:
        paths: list, list of paths to images to assign to every client
    '''
    
    images, _ = dataset.get_paths()
    
    for idx, path in enumerate(images):
        image = path.split("/")[-1]
        images[idx] = image
        
    images = np.array(images)
    np.random.shuffle(images)
    images = np.array_split(images, num_clients)
    
    
    for idx, element in enumerate(images):
        if len(element) > MAX_IMAGES_PER_CLIENT:
            trim = -1*(len(element)-MAX_IMAGES_PER_CLIENT)
            element = element[:trim]
            images[idx] = element.tolist()
    
    return images

def heterogeneous_split(dataset, num_clients):
    '''
    Heterogeneous split described in Step 3.
    Input:
        dataset: Cityscapes, the dataset to divide between clients
        num_clients: int, number of train clients to which the dataset will be divided
        
    Output:
        client_list: list, list of paths to images to assign to every client.
    '''
    
        
    images, _ = dataset.get_paths()
    np.random.shuffle(images)
    
    cities = {}
    
    for path in images:
        image = path.split("/")[-1]
        city = image.split("_")[0]
        if city not in cities:
            cities[city] = []
        
        cities[city].append(image)
    
    clients_per_city = math.floor(num_clients/len(cities))

    client_split = []
    for city in cities:
        images = np.array(cities[city])
        images = np.array_split(images, clients_per_city)
        
        for idx, element in enumerate(images):
            if len(element) > MAX_IMAGES_PER_CLIENT:
                trim = -1*(len(element)-MAX_IMAGES_PER_CLIENT)
                element = element[:trim]
            
            images[idx] = element.tolist()
        
        for image in images:
            client_split.append(image)
                
    return client_split
        