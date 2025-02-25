import matplotlib.pyplot as plt
import numpy as np

def draw_line(self, preds, pic_name, start):
    start_idx = start
    trues_plot = np.array(self.test_sequence)
    preds_plot = preds
    folder_path = f"{self.args.results_path}{self.args.model_id}/"

    plt.figure(figsize=(15, 5))
    plt.plot(trues_plot, label='Total True Values')

    plt.axhline(y=self.args.end, color='blue', linestyle='--',
                label=f'Value = {self.args.end}')
    plt.legend()
    plt.title(pic_name)
    plt.xlabel('Cycle')
    plt.ylabel('Capacity(Ah)')
    plt.grid(True)
    plt.savefig(f"{folder_path}{self.args.battery_name}.png")
    plt.show()