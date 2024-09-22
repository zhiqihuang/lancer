import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)  

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainClassifier(nn.Module):
    def __init__(self, num_layers=1, input_size=768, num_labels=2):
        super(DomainClassifier, self).__init__()
        if num_layers == 1:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, num_labels),
            )
        elif num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, num_labels),
            )
        elif num_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, num_labels),
            )
        else:
            raise NotImplementedError()

    def forward(self, inputs, labels=None):
        logits = self.classifier(inputs)

        if labels is None:
            return logits
        elif type(labels) is str:
            assert labels == 'uniform'
            return (
                logits,
                self.uniform_loss(logits),
                None,
            )
        else:
            return (
                logits,
                F.cross_entropy(logits, labels),
                self.get_acc(logits, labels)
            )
    
    @staticmethod
    def uniform_loss(logits):
        device = logits.device
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        uniform_dist = F.softmax(torch.ones_like(logits, dtype=torch.float, device=device, requires_grad=False), dim=1)
        return loss_fct(F.log_softmax(logits, dim=1), uniform_dist)
    
    @staticmethod
    def confusion_loss(logits):
        batch_size = logits.shape[0]
        device = logits.device
        return (
            F.cross_entropy(logits, torch.tensor([0] * batch_size, device=device)) + \
            F.cross_entropy(logits, torch.tensor([1] * batch_size, device=device))
        ) / 2
        
    
    @staticmethod
    def get_acc(logits, labels):
        preds = torch.argmax(logits, dim=1)
        total = int(len(labels))
        correct = int(sum(labels==preds))
        return (total, correct, correct/total)
    

class ReversalClassifier(nn.Module):
    def __init__(self, num_layers=1, input_size=768, num_labels=2):
        super(ReversalClassifier, self).__init__()
        if num_layers == 1:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, num_labels),
            )
        elif num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, num_labels),
            )
        elif num_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, input_size // 2),
                nn.ReLU(),
                nn.Linear(input_size // 2, num_labels),
            )
        else:
            raise NotImplementedError()

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs, labels, alpha=1.0):
        reversed_inputs = ReverseLayerF.apply(inputs, alpha)
        logits = self.classifier(reversed_inputs)
        loss = self.loss_fct(logits, labels)
        return loss

