import torch
from numpy import unravel_index as unravel
import matplotlib.pyplot as plt

action_dict = {'a': torch.tensor([0., -1.]), 'd': torch.tensor([0., 1.]), 'w': torch.tensor([-1., 0.]), 's': torch.tensor([1., 0.])}

def do(snake: torch.Tensor, action):
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [torch.Tensor(unravel(x, snake.shape)) for x in positions]
    #print('direction', (pos_cur - pos_prev)) # Направление движения
    pos_next = (pos_cur + action) % torch.Tensor([snake.shape]).squeeze(0) 
    
    pos_cur = pos_cur.int()
    pos_next = pos_next.int()
    
    # Проверка на столкновение
    if (snake[tuple(pos_next)] > 0).any():
        return (snake[tuple(pos_cur)] - 2).item()  # Возвращаем счёт (длина змейки минус 2)
    
    # Кушаем яблоко
    if snake[tuple(pos_next)] == -1:
        pos_food = (snake == 0).flatten().to(torch.float).multinomial(1)[0] # Генерируем позицию яблока
        snake[unravel(pos_food, snake.shape)] = -1 # Добавляем яблоко в игру
    else: # Двигаемся в пустую клетку
        snake[snake > 0] -= 1 # Уменьшаем значения

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1 # перемещаем голову
    return None

plt.rcParams['figure.figsize'] = 10, 10
snake = torch.zeros((32, 32), dtype=torch.int)
snake[1, :3] = torch.Tensor([1, 2, -1]) # [хвост, голова, яблоко]

fig, ax = plt.subplots(1, 1)
img = ax.imshow(snake)

action = {'val': 1}

n = 0
score = None
while n<10:
    img.set_data(snake)
    score = do(snake, action_dict['w'])
    n += 1