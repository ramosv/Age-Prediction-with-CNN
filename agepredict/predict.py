import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from agepredict.models import CNN, RestNet
from agepredict.utils import get_val_transforms
from pathlib import Path

class JudgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, debug=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.debug = debug
        
        # get the path from the full_path column
        def extract_path(raw):
            if isinstance(raw, str) and raw.startswith('[') and raw.endswith(']'):
                trimmed = raw.strip('[]')
                parts = trimmed.split(',')
                return parts[0].strip().strip("'").strip('"')
            return raw
        
        # extraction to the full_path column
        self.df['full_path'] = self.df['full_path'].apply(extract_path)
        
        if self.debug:
            print("Judge dataset full paths after extraction:")
            print(self.df['full_path'].head())
        
def generate_submission(judge_csv, judge_img_dir, checkpoint_path, submission_path='submission.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RestNet(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # uncomment to use local cnn model
    #model = CNN(num_classes=1).to(device)
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #model.eval()
    
    dataset = JudgeDataset(
        csv_file=judge_csv, 
        img_dir=judge_img_dir, 
        transform=get_val_transforms()
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    predictions = []
    ids = []
    
    with torch.no_grad():
        for images, batch_ids in loader:
            images = images.to(device)
            outputs = model(images)
            
            # get the numpy array from the tensor
            ages_array = outputs.cpu().numpy()

            if ages_array.ndim == 0:
                ages = [ages_array.item()]
            else:
                ages = ages_array.tolist()
            
            predictions.extend(ages)
            ids.extend(batch_ids.tolist())
    
    temp = []

    for p in predictions:
        temp.append(round(p,1))
    
    # submisison expects two columns ID and age we short by ID
    submission_df = pd.DataFrame({"ID": ids, "age": temp})
    submission_df.sort_values("ID", inplace=True)
    
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved at {submission_path}")
