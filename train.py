import torch
from torch.utils.data import DataLoader, random_split
from model import init_model
from preprocessing import load_data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(y_pred, y_true):
    predicted = torch.argmax(y_pred, dim=1)
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def main():
    dataset = load_data('data2.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ## 데이터에 맞게 dimension 수정 필요
    ## 현재 기존 데이터셋에 맞게 설정
    model = init_model(input_dim=4, hidden_dim=64, output_dim=3, num_layers=2, num_heads=4, dropout=0.3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        train_loss = 0
        train_acc = 0
        train_preds = []
        train_labels = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features.float())
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc.item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            all_preds = []
            all_labels = []
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features.float())
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)
                test_loss += loss.item()
                test_acc += acc.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            test_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1 Score: {train_f1:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}')

    save_model(model, './trained_model.pth')

if __name__ == "__main__":
    main()


