from tqdm import tqdm
import torch.nn as nn
import torch
 
def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets.view(-1,1))



def train_fn(data_loader,model,optimizer,scheduler):
    model.train()
    
    for bi ,d in tqdm(enumerate(data_loader), total = len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["target"]

        #cuda config
        """
        ids = ids.to(device,dtype = torch.long)
        mask = mask.to(device,dtype = torch.long)
        token_type_ids = token_type_ids.to(device,dtype = torch.long)
        targets = targets.to(device,dtype = torch.flaot)
        """

        optimizer.zero_grad()
        outputs = model(
            ids = ids,
            mask = mask, 
            token_type_ids = token_type_ids
        )

        loss = loss_fn(outpus,targets)
        loss.backwards()

        # if (bi+1) % accumulation_steps==0:
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader,model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi ,d in tqdm(enumerate(data_loader), total = len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            targets = d["target"]

            #cuda config
            """
            ids = ids.to(device,dtype = torch.long)
            mask = mask.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            targets = targets.to(device,dtype = torch.flaot)
            """

            # optimizer.zero_grad()
            outputs = model(
                ids = ids,
                mask = mask, 
                token_type_ids = token_type_ids
            )

            # loss = loss_fn(outpus,targets)
            loss.backwards()

            # if (bi+1) % accumulation_steps==0:
            optimizer.step()
            scheduler.step()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs,fin_targets
