import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.model = nn.Sequential(
        

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),# -> 16x224x224
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),# -> 16
            
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64
            
            
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
            
            
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256x7X7
            
            
            # Added after 40% acc
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512x7x7
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),  # -> 512x3.5X3.5
            
            
            nn.Conv2d(512, 1024, 3, padding=1),  # -> 512x7x7
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),  # -> 512x3.5X3.5
            

                         
        
            nn.Flatten(),

            nn.Linear(1024* 7 *7, 400),  # 500
            nn.Dropout(0.5),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            
#             nn.Linear(500, 250),  # 250
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(250),
#             nn.ReLU(),
            
            
            nn.Linear(400, num_classes), # 50
            
        
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"


