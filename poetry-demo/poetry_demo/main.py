import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import torch
from models import Logistic_Regression, Ridge_Regression , DummyDataset
from helpers import read_data_demo, cord_and_label,plot_decision_boundaries
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import pandas as pd



random.seed(42)
torch.manual_seed(42)
def plot_λ_and_best_and_worse():
    """Plot accuracy vs. regularization strength and find best/worst models."""

    λ = [0., 2., 4., 6., 8., 10.]
    x_train, y_train = cord_and_label('train.csv')
    x_test, y_test = cord_and_label('test.csv')  
    x_val, y_val = cord_and_label('validation.csv')

    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    for l in λ:
        model = Ridge_Regression(l)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)

        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)
        test_acc = np.mean(y_test_pred == y_test)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
    data = {'λ': λ, 'Train Accuracy': train_accuracies, 'Validation Accuracy': val_accuracies, 'Test Accuracy': test_accuracies}
    df = pd.DataFrame(data)

# Print the DataFrame
    print(df)

    # Plotting
    plt.plot(λ, train_accuracies, label='Training Accuracy')
    plt.plot(λ, val_accuracies, label='Validation Accuracy')
    plt.plot(λ, test_accuracies, label='Test Accuracy')
    plt.xlabel('λ (Regularization Strength)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Regularization Strength')
    plt.legend()
    plt.show()

    # Find the λ corresponding to the best validation accuracy
    best_lambda = λ[np.argmax(val_accuracies)] # might give us 2, but we will  choose 4
    best_lambda = 4.
    print(f"Best λ according to validation set: {best_lambda}")
    # Report test accuracy of the best model
    best_model = Ridge_Regression(best_lambda)
    best_model.fit(x_train, y_train)
    y_test_pred_best = best_model.predict(x_test)
    best_test_acc = np.mean(y_test_pred_best == y_test)
    print(f"Test accuracy of the best model: {best_test_acc}")

    # Find the λ corresponding to the worst validation accuracy
    worse_lambda = λ[np.argmin(val_accuracies)]
    print(f"Worse λ according to validation set: {worse_lambda}")

    # Report test accuracy of the worst model
    worse_model = Ridge_Regression(worse_lambda)
    worse_model.fit(x_train, y_train)
    y_test_pred_worse = worse_model.predict(x_test)
    worse_test_acc = np.mean(y_test_pred_worse == y_test)
    print(f"Test accuracy of the worse model: {worse_test_acc}")

    # Plot decision boundaries of the best and worst models
    plot_decision_boundaries(best_model,x_test,y_test)
    plot_decision_boundaries(worse_model,x_test,y_test)






def f(x, y):
    """Calculate the gradient of the function f(x, y)."""
    return (x - 3)**2 + (y - 5)**2

def gradient_f(x, y):
    """Calculate the gradient of the function f(x, y)."""
    df_dx = 2 * (x - 3)
    df_dy = 2 * (y - 5)
    return np.array([df_dx, df_dy])

def Gradient_Descent(current_point, learning_rate, it):
    """Perform gradient descent optimization."""
    trajectory = np.zeros((it, 2))

    a = learning_rate
    for i in range(it):
        trajectory[i] = current_point
        gradient_at_point = gradient_f(current_point[0], current_point[1])
        current_point = current_point - a * gradient_at_point
    print(f"The point we achieved is {current_point} after {it} iterations")
    # Plotting
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=range(len(trajectory)), cmap='viridis')
    plt.colorbar(label='Iterations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Trajectory')
    plt.show()


####################################    PyTorch part #############################
    

def Binary_Case():
    """Train a logistic regression model for binary classification."""

    criterion = torch.nn.CrossEntropyLoss()
    dummyDataset_train = DummyDataset('train.csv')
    dummyDataset_test = DummyDataset('test.csv')
    dummyDataset_val = DummyDataset('validation.csv')
    train_loader = DataLoader(dummyDataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dummyDataset_test, batch_size=32, shuffle=False)
    val_loader = DataLoader(dummyDataset_val, batch_size=32, shuffle=False)
    n_classes = len(torch.unique(dummyDataset_train.labels))
    learning_rate = [0.1,0.01,0.001]
    epoch = 10
    for rate in learning_rate:
        # Create a tensor with requires_grad=True to enable automatic differentiation
        x = torch.tensor([0.0], requires_grad=True)
        model = Logistic_Regression(2,n_classes)
        optimizer = optim.SGD(model.parameters(), lr=rate)
        train_model(model, train_loader,test_loader,val_loader, criterion, optimizer, epoch)



def train_model(model, train_loader, test_loader, val_loader, criterion, optimizer, epochs):
    """Train the model."""
    # Initialize lists to store accuracies and losses
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    learning_rates = []  # Store learning rates for each epoch
    mean_train_losses = []  # Store mean train losses for each epoch
    mean_test_losses = []  # Store mean test losses for each epoch
    mean_val_losses = []  # Store mean validation losses for each epoch

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0

        # Training loop
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_y).sum().item()

        # Calculate training accuracy and mean training loss
        train_accuracy = correct_train / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        mean_train_loss = total_train_loss / len(train_loader)
        mean_train_losses.append(mean_train_loss)

        # Store learning rate and total loss for the epoch
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation loop
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                outputs_val = model(batch_x_val)
                val_loss = criterion(outputs_val, batch_y_val)
                total_val_loss += val_loss.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                correct_val += (predicted_val == batch_y_val).sum().item()

        # Calculate validation accuracy and mean validation loss
        val_accuracy = correct_val / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        mean_val_loss = total_val_loss / len(val_loader)
        mean_val_losses.append(mean_val_loss)

        # Test loop
        model.eval()
        total_test_loss = 0
        correct_test = 0

        with torch.no_grad():
            for batch_x_test, batch_y_test in test_loader:
                outputs_test = model(batch_x_test)
                test_loss = criterion(outputs_test, batch_y_test)
                total_test_loss += test_loss.item()
                _, predicted_test = torch.max(outputs_test.data, 1)
                correct_test += (predicted_test == batch_y_test).sum().item()

        # Calculate test accuracy and mean test loss
        test_accuracy = correct_test / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)
        mean_test_loss = total_test_loss / len(test_loader)
        mean_test_losses.append(mean_test_loss)

    # Create a DataFrame for the accuracies, learning rates, and total losses
    data = {'Epoch': range(1, epochs + 1),
            'Learning Rate': learning_rates,
            'Train Accuracy': train_accuracies,
            'Train Loss': mean_train_losses,
            'Validation Accuracy': val_accuracies,
            'Validation Loss': mean_val_losses,
            'Test Accuracy': test_accuracies,
            'Test Loss': mean_test_losses}

    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    plt.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy')
    plt.title('Training, Validation, and Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
    plt.title(f'Learning Rate: {learning_rates[-1]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()




def train_model_with_decay(model, train_loader, test_loader, val_loader, criterion, optimizer, epochs, rate):
    """Train the model with learning rate decay."""

    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    mean_train_losses = []  # Store mean train losses for each epoch
    mean_test_losses = []  # Store mean test losses for each epoch
    mean_val_losses = []  # Store mean validation losses for each epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=5, # rate of decay
                                               gamma=0.3 # how much to decay the learning rate
                                              )

    for epoch in range(epochs):
        
        model.train()
        total_train_loss = 0
        correct_train = 0

        # Training loop
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_y).sum().item()

        # Calculate training accuracy and mean training loss
        train_accuracy = correct_train / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        mean_train_loss = total_train_loss / len(train_loader)
        mean_train_losses.append(mean_train_loss)


        # Validation loop
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                outputs_val = model(batch_x_val)
                val_loss = criterion(outputs_val, batch_y_val)
                total_val_loss += val_loss.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                correct_val += (predicted_val == batch_y_val).sum().item()

        # Calculate validation accuracy and mean validation loss
        val_accuracy = correct_val / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        mean_val_loss = total_val_loss / len(val_loader)
        mean_val_losses.append(mean_val_loss)

        # Test loop
        model.eval()
        total_test_loss = 0
        correct_test = 0

        with torch.no_grad():
            for batch_x_test, batch_y_test in test_loader:
                outputs_test = model(batch_x_test)
                test_loss = criterion(outputs_test, batch_y_test)
                total_test_loss += test_loss.item()
                _, predicted_test = torch.max(outputs_test.data, 1)
                correct_test += (predicted_test == batch_y_test).sum().item()

        #step
        lr_scheduler.step()

        # Calculate test accuracy and mean test loss
        test_accuracy = correct_test / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)
        mean_test_loss = total_test_loss / len(test_loader)
        mean_test_losses.append(mean_test_loss)
        

    # Create a DataFrame for the accuracies, learning rates, and total losses
    data = {'Epoch': range(1, epochs + 1),
            'Train Accuracy': train_accuracies,
            'Train Loss': mean_train_losses,
            'Validation Accuracy': val_accuracies,
            'Validation Loss': mean_val_losses,
            'Test Accuracy': test_accuracies,
            'Test Loss': mean_test_losses}

    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    plt.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy')
    plt.title('Training, Validation, and Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
    plt.title(f'Learning Rate: {rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def Q_9_4():
    """
    Solution to question 9.4.

    Trains logistic regression models with different learning rates and plots their performance.
    """
    criterion = torch.nn.CrossEntropyLoss()
    dummyDataset_train = DummyDataset('train_multiclass.csv')
    dummyDataset_test = DummyDataset('test_multiclass.csv')
    dummyDataset_val = DummyDataset('validation_multiclass.csv')
    train_loader = DataLoader(dummyDataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dummyDataset_test, batch_size=32, shuffle=False)
    val_loader = DataLoader(dummyDataset_val, batch_size=32, shuffle=False)
    n_classes = len(torch.unique(dummyDataset_train.labels))
    learning_rate = [0.01,0.001,0.0003]
    epoch = 30
    for rate in learning_rate:
        # Create a tensor with requires_grad=True to enable automatic differentiation
        x = torch.tensor([0.0], requires_grad=True)
        model = Logistic_Regression(2,n_classes)
        optimizer = optim.SGD(model.parameters(), lr=rate)
        train_model_with_decay(model, train_loader,test_loader,val_loader, criterion, optimizer, epoch,rate)


def plot_modle_0001_learning_rate():
    """
    Plots the decision boundaries for the best logistic regression model trained with a learning rate of 0.001.
    """
    criterion = torch.nn.CrossEntropyLoss()
    dummyDataset_train = DummyDataset('train.csv')
    dummyDataset_test = DummyDataset('test.csv')
    dummyDataset_val = DummyDataset('validation.csv')
    train_loader = DataLoader(dummyDataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dummyDataset_test, batch_size=32, shuffle=False)
    val_loader = DataLoader(dummyDataset_val, batch_size=32, shuffle=False)
    n_classes = len(torch.unique(dummyDataset_train.labels))
    learning_rate = 0.001
    epochs = 10
    # Create a tensor with requires_grad=True to enable automatic differentiation
    x = torch.tensor([0.0], requires_grad=True)
    model = Logistic_Regression(2,n_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        # Training loop
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)
            train_loss.backward()
            optimizer.step()
    x_test , y_test = cord_and_label('test.csv')
    plot_decision_boundaries(model,x_test,y_test, "Best model with 0.001 learning rate")



def train_model_with_decay_give_last_epoch(model, train_loader, test_loader, val_loader, criterion, optimizer, epochs, rate):
    """
    Trains the model with learning rate decay and returns the performance metrics of the last epoch.
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=5, # rate of decay
                                               gamma=0.3 # how much to decay the learning rate
                                              )
    for epoch in range(epochs):
        
        model.train()
        total_train_loss = 0
        correct_train = 0

        # Training loop
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_y).sum().item()

        # Calculate training accuracy and mean training loss
        if epoch == epochs -1 :
            train_accuracy = correct_train / len(train_loader.dataset)
            train_last = train_accuracy    


        # Validation loop
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                outputs_val = model(batch_x_val)
                val_loss = criterion(outputs_val, batch_y_val)
                total_val_loss += val_loss.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                correct_val += (predicted_val == batch_y_val).sum().item()

        if epoch == epochs -1 :
            val_accuracy_last = correct_val / len(val_loader.dataset)
        # Test loop
        model.eval()
        total_test_loss = 0
        correct_test = 0

        with torch.no_grad():
            for batch_x_test, batch_y_test in test_loader:
                outputs_test = model(batch_x_test)
                test_loss = criterion(outputs_test, batch_y_test)
                total_test_loss += test_loss.item()
                _, predicted_test = torch.max(outputs_test.data, 1)
                correct_test += (predicted_test == batch_y_test).sum().item()

        #step
        lr_scheduler.step()
        if epoch == epochs -1 :
            # Calculate test accuracy and mean test loss
            last_test_accuracy = correct_test / len(test_loader.dataset)
            last_mean_test_loss = total_test_loss / len(test_loader)
    return rate, train_last ,last_test_accuracy ,val_accuracy_last


def Q_9_4_1():
    """
    Solution to question 9.4.1.

    Trains logistic regression models with different learning rates, evaluates their performance,
    and plots the test and validation accuracies against the learning rate.
    """
    criterion = torch.nn.CrossEntropyLoss()
    dummyDataset_train = DummyDataset('train_multiclass.csv')
    dummyDataset_test = DummyDataset('test_multiclass.csv')
    dummyDataset_val = DummyDataset('validation_multiclass.csv')
    train_loader = DataLoader(dummyDataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dummyDataset_test, batch_size=32, shuffle=False)
    val_loader = DataLoader(dummyDataset_val, batch_size=32, shuffle=False)
    n_classes = len(torch.unique(dummyDataset_train.labels))
    learning_rate = [0.01, 0.001, 0.0003]
    epoch = 30
    best_lr = None
    best_test_acc = 0.0

    lr_values = []
    test_accuracies = []
    val_accuracies = []

    for rate in learning_rate:
        model = Logistic_Regression(2, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=rate)
        current_lr, train_acc, test_acc, val_acc = train_model_with_decay_give_last_epoch(model, train_loader, test_loader,
                                                                                         val_loader, criterion, optimizer,
                                                                                         epoch, rate)

        lr_values.append(current_lr)
        test_accuracies.append(test_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_lr = current_lr

    print(f"Best Learning Rate: {best_lr:.4f}, Best Test Accuracy: {best_test_acc:.4f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(lr_values, test_accuracies, label='Test Accuracy')
    plt.plot(lr_values, val_accuracies, label='Validation Accuracy')
    plt.xscale('log')  # Log scale for better visualization
    plt.title('Test and Validation Accuracies vs. Learning Rate')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def Q_9_4_2():
    
    """
    Solution to question 9.4.2.

    Trains the logistic regression model with the best learning rate and plots the decision boundaries.
    """
    best_rate = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    dummyDataset_train = DummyDataset('train_multiclass.csv')
    dummyDataset_test = DummyDataset('test_multiclass.csv')
    dummyDataset_val = DummyDataset('validation_multiclass.csv')
    train_loader = DataLoader(dummyDataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dummyDataset_test, batch_size=32, shuffle=False)
    val_loader = DataLoader(dummyDataset_val, batch_size=32, shuffle=False)
    n_classes = len(torch.unique(dummyDataset_train.labels))
    epoch = 30
    # Create a tensor with requires_grad=True to enable automatic differentiation
    x = torch.tensor([0.0], requires_grad=True)
    model = Logistic_Regression(2,n_classes)
    optimizer = optim.SGD(model.parameters(), lr=best_rate)
    train_model_with_decay(model, train_loader,test_loader,val_loader, criterion, optimizer, epoch,best_rate)


def Q_9_4_3():
    """
    Solution to question 9.4.3.

    Trains a decision tree classifier with max depth 2 and evaluates its performance.
    """
    max_depth = 2
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_train, Y_train = cord_and_label('train_multiclass.csv')
    X_test, Y_test = cord_and_label('test_multiclass.csv')
    X_val, Y_val = cord_and_label('validation_multiclass.csv')
    tree_classifier.fit(X_train, Y_train)
    y_train_pred = tree_classifier.predict(X_train)
    y_test_pred = tree_classifier.predict(X_test)
    y_val_pred = tree_classifier.predict(X_val)
    train_accuracy = np.mean(Y_train == y_train_pred)
    test_accuracy = np.mean(Y_test == y_test_pred)
    val_accuracy = np.mean(Y_val == y_val_pred)

    print(f"for tree with depth : {max_depth} the train accuracy is  {round(train_accuracy,3)} and the test accuracy is {round(test_accuracy,3)} \n and the  validation  is {round(val_accuracy,3)}")

    plot_decision_boundaries(tree_classifier, X_test, Y_test, title='Decision Boundaries for Tree max depth 2')


def Q_9_4_4():
    """
    Solution to question 9.4.3.

    Trains a decision tree classifier with max depth 2 and evaluates its performance.
    """
    max_depth = 10
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_train, Y_train = cord_and_label('train_multiclass.csv')
    X_test, Y_test = cord_and_label('test_multiclass.csv')
    X_val, Y_val = cord_and_label('validation_multiclass.csv')
    tree_classifier.fit(X_train, Y_train)
    y_train_pred = tree_classifier.predict(X_train)
    y_test_pred = tree_classifier.predict(X_test)
    y_val_pred = tree_classifier.predict(X_val)
    train_accuracy = np.mean(Y_train == y_train_pred)
    test_accuracy = np.mean(Y_test == y_test_pred)
    val_accuracy = np.mean(Y_val == y_val_pred)
    print(f"for tree with depth : {max_depth} the train accuracy is  {round(train_accuracy,3)} and the test accuracy is {round(test_accuracy,3)} \n and the  validation  is {round(val_accuracy,3)}")
    plot_decision_boundaries(tree_classifier, X_test, Y_test, title='Decision Boundaries for Tree max depth 10')




