import os
import torch
import random
import numpy as np
import pandas as pd


def save_predictions_to_excel(model_id, root_path, data_path):
    folder_path = os.path.join('./results', model_id)
    short_name = 'compare'

    preds_file_path = os.path.join(folder_path, 'pred.npy')
    preds = np.load(preds_file_path).flatten()

    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    num_rows = len(df_raw)
    preds_len = len(preds)
    twenty_percent_index = num_rows - preds_len
    df_raw = df_raw.iloc[twenty_percent_index:]
    cols_data = df_raw.columns[1:]
    trues = df_raw[cols_data].to_numpy().flatten()

    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-2] * preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2] * trues.shape[-1])

    df = pd.DataFrame({
        'True Values': trues,
        'Predicted Values': preds
    })

    excel_path = os.path.join(folder_path, f'{short_name}.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Predictions and true values saved to {excel_path}")


def set_seed(seed):
    random.seed(seed)
    seed += 1
    np.random.seed(seed)
    seed += 1
    torch.manual_seed(seed)


battery_EOL = {
    'Cell_1': [100],
    'Cell_2': [100],
    'Cell_3': [100],
    'Cell_4': [100]
}
train_battery = 'Cell_1'


class Args:
    def __init__(self):
        self.model_id = f'ETSformer#{train_battery}'
        self.model = 'ETSformer'
        self.data = 'Battery'
        self.end = 0.1

        self.battery_EOL = battery_EOL
        self.battery_name = f'Cell_{train_battery}'
        self.root_path = './datasets/'
        self.results_path = './results/'
        self.MVMD_path = './datasets/'
        self.data_path = f'battery_data_frames[{self.battery_name}].csv'

        self.checkpoints = './ETS_checkpoints'

        self.seq_len = 300
        self.batch_size = 32
        self.patience = 5000
        self.epochs = 20
        self.learning_rate = 1e-4
        self.d_model = 256
        self.dropout = 0.01
        self.d_ff = 4096
        self.num_workers = 10
        self.n_heads = 2
        self.e_layers = 2
        self.d_layers = 2
        self.warmup_epochs = 2
        self.pred_len = 1

        self.label_len = 10
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.activation = 'sigmoid'
        self.min_lr = 2e-30
        self.std = 0.2
        self.smoothing_learning_rate = 0
        self.damping_learning_rate = 0
        self.output_attention = True
        self.optim = 'adam'
        self.itr = 1
        self.des = 'test'
        self.lradj = 'exponential_with_warmup'
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'


args = Args()

args.use_gpu = torch.cuda.is_available() and args.use_gpu

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = [int(id_) for id_ in args.devices.split(',')]
    args.device_ids = device_ids
    args.gpu = args.device_ids[0]

setting = (
    f"{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_bz{args.batch_size}_pc{args.patience}"
    f"_te{args.epochs}_lr{args.learning_rate}_dm{args.d_model}_dp{args.dropout}"
    f"_df{args.d_ff}_nh{args.n_heads}_ly{args.e_layers}"
)

exp = Exp(args)  # set experiments

print('>>>>>>>>>>>>>>>>>>>>>>>>>> >start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()
# exp.test(write=False)
print("------------------------------ train end ------------------------------")

torch.cuda.empty_cache()