import sys
import matplotlib.pyplot as plt



def plot_los_val_loss(train_info):
    plt.plot(train_info.keys(),
             [loss['loss'] for _, loss in train_info.items()],
             'r',
             label='loss')
    plt.plot(train_info.keys(),
             [val_loss['val_loss'] for _, val_loss in train_info.items()],
             'b',
             label='validation loss')
    plt.legend(loc="upper right")

    plt.show()


def get_train_data_from_txt(file_name):
    with open(file_name, mode="r", newline=None) as input_file:
        input_data = [x.strip() for x in input_file.readlines()]

    data_dict = {}
    loss_no = -1
    val_loss_no = -1
    epoch_no = -1

    for el in input_data:
        if 'Epoch' in el:
            epoch_no = el[el.find(' '):el.find('/')].strip()
            loss_no = -1
            val_loss_no = -1
        if 'loss' in el:
            loss_no = el[el.find('loss:') + len('loss:'):]
            loss_no = float(loss_no[:loss_no.find('.') + 4].strip())
        if 'val_loss' in el:
            val_loss_no = el[el.find('val_loss') + len('val_loss:'):].strip()
            val_loss_no = float(val_loss_no)
        if loss_no != -1 and val_loss_no != -1 and epoch_no != -1:
            data_dict[epoch_no] = dict(loss=loss_no, val_loss=val_loss_no)
            epoch_no = -1

    return data_dict
  
