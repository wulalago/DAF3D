from DAF3D import DAF3D
from DataOperate import MySet, get_data_list
from Utils import DiceLoss, dice_ratio


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import time
import os


if __name__ == '__main__':

    train_list, test_list = get_data_list("Data/Original", ratio=0.8)

    best_dice = 0.

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    information_line = '='*20 + ' DAF3D ' + '='*20 + '\n'
    open('Log.txt', 'w').write(information_line)

    torch.cuda.set_device(0)
    net = DAF3D().cuda()

    criterion_bce = torch.nn.BCELoss()
    criterion_dice = DiceLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    train_set = MySet(train_list)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_dataset = MySet(test_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for epoch in range(1, 21):
        epoch_start_time = time.time()
        print("Epoch: {}".format(epoch))
        epoch_loss = 0.
        net.train()
        start_time = time.time()
        for batch_idx, (image, label) in enumerate(train_loader):
            image = Variable(image.cuda())
            label = Variable(label.cuda())

            optimizer.zero_grad()

            outputs1, outputs2, outputs3, outputs4, outputs1_1, outputs1_2, outputs1_3, outputs1_4, output = net(image)

            output = F.sigmoid(output)
            outputs1 = F.sigmoid(outputs1)
            outputs2 = F.sigmoid(outputs2)
            outputs3 = F.sigmoid(outputs3)
            outputs4 = F.sigmoid(outputs4)
            outputs1_1 = F.sigmoid(outputs1_1)
            outputs1_2 = F.sigmoid(outputs1_2)
            outputs1_3 = F.sigmoid(outputs1_3)
            outputs1_4 = F.sigmoid(outputs1_4)

            loss0_bce = criterion_bce(output, label)
            loss1_bce = criterion_bce(outputs1, label)
            loss2_bce = criterion_bce(outputs2, label)
            loss3_bce = criterion_bce(outputs3, label)
            loss4_bce = criterion_bce(outputs4, label)
            loss5_bce = criterion_bce(outputs1_1, label)
            loss6_bce = criterion_bce(outputs1_2, label)
            loss7_bce = criterion_bce(outputs1_3, label)
            loss8_bce = criterion_bce(outputs1_4, label)

            loss0_dice = criterion_dice(output, label)
            loss1_dice = criterion_dice(outputs1, label)
            loss2_dice = criterion_dice(outputs2, label)
            loss3_dice = criterion_dice(outputs3, label)
            loss4_dice = criterion_dice(outputs4, label)
            loss5_dice = criterion_dice(outputs1_1, label)
            loss6_dice = criterion_dice(outputs1_2, label)
            loss7_dice = criterion_dice(outputs1_3, label)
            loss8_dice = criterion_dice(outputs1_4, label)

            loss = loss0_bce + 0.4 * loss1_bce + 0.5 * loss2_bce + 0.7 * loss3_bce + 0.8 * loss4_bce + \
                   0.4 * loss5_bce + 0.5 * loss6_bce + 0.7 * loss7_bce + 0.8 * loss8_bce + \
                   loss0_dice + 0.4 * loss1_dice + 0.5 * loss2_dice + 0.7 * loss3_dice + 0.8 * loss4_dice + \
                   0.4 * loss5_dice + 0.7 * loss6_dice + 0.8 * loss7_dice + 1 * loss8_dice

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print_line = 'Epoch: {} | Batch: {} -----> Train loss: {:4f} Cost Time: {}\n' \
                             'Batch bce  Loss: {:4f} || ' \
                             'Loss1: {:4f}, Loss2: {:4f}, Loss3: {:4f}, Loss4: {:4f}, ' \
                             'Loss5: {:4f}, Loss6: {:4f}, Loss7: {:4f}, Loss8: {:4f}\n' \
                             'Batch dice Loss: {:4f} || ' \
                             'Loss1: {:4f}, Loss2: {:4f}, Loss3: {:4f}, Loss4: {:4f}, ' \
                             'Loss5: {:4f}, Loss6: {:4f}, Loss7: {:4f}, Loss8: {:4f}\n' \
                    .format(epoch, batch_idx, epoch_loss / (batch_idx + 1), time.time() - start_time,
                            loss0_bce.item(), loss1_bce.item(), loss2_bce.item(), loss3_bce.item(), loss4_bce.item(),
                            loss5_bce.item(), loss6_bce.item(), loss7_bce.item(), loss8_bce.item(),
                            loss0_dice.item(), loss1_dice.item(), loss2_dice.item(), loss3_dice.item(),
                            loss4_dice.item(), loss5_dice.item(), loss6_dice.item(), loss7_dice.item(),
                            loss8_dice.item())
                print(print_line)
                start_time = time.time()

            loss.backward()
            optimizer.step()

        print('Epoch {} Finished ! Loss is {:4f}'.format(epoch, epoch_loss / (batch_idx + 1)))
        open('Log.txt', 'a') \
            .write("Epoch {} Loss: {}".format(epoch, epoch_loss / (batch_idx + 1)))

        print("Epoch time: ", time.time() - epoch_start_time)
        # begin to eval
        net.eval()

        dice = 0.

        for batch_idx, (image, label) in enumerate(test_loader):
            image = Variable(image.cuda())
            label = Variable(label.cuda())

            predict = net(image)
            predict = F.sigmoid(predict)

            predict = predict.data.cpu().numpy()
            label = label.data.cpu().numpy()

            dice_tmp = dice_ratio(predict, label)
            dice = dice + dice_tmp

        dice = dice / (1 + batch_idx)
        print("Eva Dice Result: {}".format(dice))
        open('Log.txt', 'a').write("Epoch {} Dice Score: {}\n".format(epoch, dice))

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), 'checkpoints/Best_Dice.pth')

        torch.save(net.state_dict(), 'checkpoints/model_{}.pth'.format(epoch))
