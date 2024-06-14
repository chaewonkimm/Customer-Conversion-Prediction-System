#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def predict(model, data_loader):
    model.eval()
    predicted_probabilities = []
    with torch.no_grad():
        for features, _ in data_loader:
            logits = model(features)
            probabilities = F.softmax(logits, dim=1)
            predicted_probabilities.extend(probabilities.tolist())
    return predicted_probabilities

if __name__ == "__main__":
    # Assuming load_data is defined elsewhere and correctly implemented
    dataset = load_data('data_7(수정).csv')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = init_model(input_dim=9, hidden_dim=50, output_dim=2, num_layers=2, num_heads=2, dropout=0.3)

    model = load_model(model, './trained_model_leave2(7).pth')

    predictions = predict(model, data_loader)

    print("predict finish")

