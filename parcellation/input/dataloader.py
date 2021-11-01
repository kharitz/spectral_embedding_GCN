import torch.utils.data
from parcellation.input.batch import Batch


class DataLoader(torch.utils.data.DataLoader):
    """Data loader which reads all the graph information such as coordinates and faces.
    As of now implemented to just as a loader
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): Default set to 1 as the brain graphs are large
        shuffle (bool, optional): `True` or 'Flase': Self explanatory. Default is 'False'            
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 follow_batch=[]):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: Batch.from_data_list(batch))
