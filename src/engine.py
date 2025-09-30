from src.utils import plot_graph, show_incorrect_images
import torch

from tqdm.auto import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train_step(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc = 100*correct/processed
  return train_loss, train_acc

def test_step(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0
    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1)
            correct_mask = pred.eq(target)
            incorrect_indices = ~correct_mask

            test_incorrect_pred['images'].extend(data[incorrect_indices])
            test_incorrect_pred['ground_truths'].extend(target[incorrect_indices])
            test_incorrect_pred['predicted_vals'].extend(pred[incorrect_indices])

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)


    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc, test_incorrect_pred

def train(model, train_loader, test_loader, device, optimizer, epochs, criterion, scheduler):

    class_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    # Data to plot accuracy and loss graphs
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        train_loss, train_acc = train_step(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
        test_loss, test_acc, test_incorrect_pred = test_step(model=model, device=device, test_loader=test_loader, criterion=criterion)
        scheduler.step()

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    plot_graph(results["train_loss"], results["test_loss"], results["train_acc"], results["test_acc"])
    show_incorrect_images(test_incorrect_pred, class_map)

    # Return the filled results at the end of the epochs
    return results
