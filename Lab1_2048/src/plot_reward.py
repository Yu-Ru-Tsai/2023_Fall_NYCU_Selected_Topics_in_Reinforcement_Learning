import matplotlib.pyplot as plt

if __name__ =='__main__':
    file_path = "./reward.txt"  
    data = []  

    with open(file_path, "r") as file:
        for line in file:
            data.append(float(line.strip()))
    plt.figure(figsize=(8, 6))  
    x = range(1000, len(data)*1000 + 1000, 1000)

    plt.plot(x, data, linestyle='-', color='b', label='score')

    # 添加标题和标签
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig('score')