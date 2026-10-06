import torch  # імпортуємо PyTorch
from torch import nn  # імпортуємо модулі для створення нейронної мережі
import torch.nn.functional as F  # імпортуємо готові функції, зокрема cross-entropy


def torch_reference(X, y, W1, b1, W2, b2):
    model = nn.Sequential(  # створюємо таку саму мережу, як у NumPy
        nn.Linear(4, 8),  # перший лінійний шар: 4 входи → 8 нейронів
        nn.ReLU(),  # застосовуємо ReLU після першого шару
        nn.Linear(8, 3),  # другий лінійний шар: 8 входів → 3 класи
    ).double()  # переводимо всі параметри моделі у float64

    with torch.no_grad():  # копіюємо параметри без запису цих операцій в autograd-граф
        model[0].weight.copy_(torch.from_numpy(W1.T))  # копіюємо W1, транспонуючи під формат PyTorch
        model[0].bias.copy_(torch.from_numpy(b1))  # копіюємо bias першого шару

        model[2].weight.copy_(torch.from_numpy(W2.T))  # копіюємо W2, транспонуючи під формат PyTorch
        model[2].bias.copy_(torch.from_numpy(b2))  # копіюємо bias другого шару

    X_t = torch.from_numpy(X)  # перетворюємо NumPy-матрицю ознак у PyTorch tensor
    y_t = torch.from_numpy(y).long()  # перетворюємо класи у tensor цілих чисел

    model.zero_grad()  # обнуляємо попередньо накопичені градієнти

    logits_t = model(X_t)  # виконуємо forward і отримуємо logits

    loss_t = F.cross_entropy(  # обчислюємо cross-entropy через PyTorch
        logits_t,  # передаємо сирі logits моделі
        y_t,  # передаємо правильні класи
        reduction="mean",  # усереднюємо loss по всіх об’єктах
    )

    loss_t.backward()  # autograd автоматично обчислює всі градієнти

    dW1_t = model[0].weight.grad.T.detach().numpy()  # беремо dW1, транспонуємо у формат NumPy
    db1_t = model[0].bias.grad.detach().numpy()  # беремо градієнт за b1

    dW2_t = model[2].weight.grad.T.detach().numpy()  # беремо dW2, транспонуємо у формат NumPy
    db2_t = model[2].bias.grad.detach().numpy()  # беремо градієнт за b2

    return (  # повертаємо еталонні результати PyTorch
        loss_t.item(),  # перетворюємо loss з tensor у звичайне число
        dW1_t,  # градієнт за W1
        db1_t,  # градієнт за b1
        dW2_t,  # градієнт за W2
        db2_t,  # градієнт за b2
    )