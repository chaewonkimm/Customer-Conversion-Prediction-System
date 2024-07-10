import torch
from torch.utils.data import DataLoader, random_split
from model import init_model
from preprocessing import load_data
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score

def calculate_accuracy(y_pred, y_true):
    predicted = torch.argmax(y_pred, dim=1)
    correct = (predicted == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_data('구매전환율_data(최종).csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = init_model(input_dim=128, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        all_train_labels = []
        all_train_preds = []
        
        for x_second, x_minute, x_hour, labels in train_loader:
            x_second, x_minute, x_hour, labels = x_second.to(device), x_minute.to(device), x_hour.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x_second, x_minute, x_hour)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        train_acc = correct / total
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            all_test_labels = []
            all_test_preds = []
            
            for x_second, x_minute, x_hour, labels in test_loader:
                x_second, x_minute, x_hour, labels = x_second.to(device), x_minute.to(device), x_hour.to(device), labels.to(device)
                outputs = model(x_second, x_minute, x_hour)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())

            test_acc = correct / total
            test_recall = recall_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
            test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0)
            test_loss /= len(test_loader)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}')
        print(f'Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

    save_model(model, './trained_model_purchase.pth')

if __name__ == "__main__":
    main()
