# Tool/EarlyStopping.py
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
    
    def check(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            
        return False