import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('pose_data.csv')
x = df['field.position.x']
y = df['field.position.y']
plt.figure()
plt.plot(x,y,linestyle='--')
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.gca().set_aspect('equal')
plt.show()
