import torch
import sys
import os


def test(checkpoint,
         criterion,
         dataset,
         debug,
         device,
         metrics):
    checkpoint.model.eval()
    with torch.no_grad():
        results = [0 for _ in range(len(metrics))]
        global_loss = 0
        for i, (data, target) in enumerate(dataset['test']):
            if debug:
                if i != 0:
                    break
            data, target = data.to(device), target.to(device)

            output = checkpoint.model(data)
            loss = criterion(output, target.long())
            for k, m in enumerate(metrics):
                results[k] += m(output, target)
            global_loss += loss.item()

            message = f'\rTest ({i + 1}/{len(dataset["test"])}) -> '
            message += f'{criterion.name} loss = {round(float(global_loss) / (i + 1), 3)}, '
            for k, m in enumerate(metrics):
                message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["test"].batch_size), 3)}\t'
            message += '           '
            sys.stdout.write(message)
    return global_loss, results


def train(checkpoint,
          criterion,
          dataset,
          debug,
          device,
          metrics):
    results = [0 for _ in range(len(metrics))]
    global_loss = 0
    checkpoint.model.train()
    for i, (data, target) in enumerate(dataset['train']):
        if debug:
            if i != 0:
                break
        data, target = data.to(device), target.to(device)
        checkpoint.optimizer.zero_grad()
        output = checkpoint.model(data)
        loss = criterion(output, target.long())
        loss.backward()
        checkpoint.optimizer.step()
        for k, m in enumerate(metrics):
            results[k] += m(output, target)
        global_loss += loss.item()

        message = f'\rTrain ({i + 1}/{len(dataset["train"])}) -> '
        message += f'{criterion.name} loss = {round(float(global_loss) / (i + 1), 3)}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {round(float(results[k]) / ((i + 1) * dataset["train"].batch_size), 3)}\t'
        message += '           '
        sys.stdout.write(message)


def save_results(results_file_name,
                 message_head,
                 criterion,
                 dataset,
                 current_epoch,
                 epochs,
                 global_loss,
                 metrics,
                 output_path,
                 results):
    if not os.path.isdir(output_path):
        try:
            os.mkdir(output_path)
        except OSError:
            print(f'Failed to create the folder {output_path}')
        else:
            print(f'Created folder {output_path}')
    with open(os.path.join(output_path, results_file_name + '_results.txt'), 'a') as f:
        message = f'{message_head}: epoch {current_epoch}/{epochs} -> '
        loss = float(global_loss) / (len(dataset["test"]) * dataset["test"].batch_size)
        message += f'{criterion.name} loss = {loss}, '
        for k, m in enumerate(metrics):
            message += f'{m.name} = {float(results[k]) / (len(dataset["test"]) * dataset["test"].batch_size)} '
        message += '\n'
        f.write(message)


def train_model(checkpoint,
                dataset,
                epochs,
                output_path,
                debug,
                criterion,
                device,
                metrics,
                name):
    e = 0
    while e < epochs:
        e = checkpoint.store_model(e)
        if e >= epochs:
            break
        e += 1
        print(f'\nEpoch {e}/{epochs}')

        train(
            checkpoint=checkpoint,
            criterion=criterion,
            dataset=dataset,
            debug=debug,
            device=device,
            metrics=metrics)

        print()
        global_loss, results = test(
            checkpoint=checkpoint,
            criterion=criterion,
            dataset=dataset,
            debug=debug,
            device=device,
            metrics=metrics)

        save_results(
            results_file_name=checkpoint.name,
            message_head=name,
            criterion=criterion,
            dataset=dataset,
            current_epoch=e,
            epochs=epochs,
            global_loss=global_loss,
            metrics=metrics,
            output_path=output_path,
            results=results)

        checkpoint.scheduler.step()

    checkpoint.store_model(output_path, epochs)
