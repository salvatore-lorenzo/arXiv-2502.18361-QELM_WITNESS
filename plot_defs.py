
import matplotlib.pyplot as plt
import numpy as np

def witness_plot(title,x_train,y_train,x_test,y_test,mse,ax,kk,labels,col):
    band_height = 3*np.sqrt(mse)  
    ax.axhspan(-band_height/2, band_height/2, color='gray', alpha=0.2)  
    ax.axhline(-band_height/2, color='gray', linewidth=1.5, linestyle='-.')
    ax.axhline(band_height/2, color='gray', linewidth=1.5, linestyle='-.')
    ll=0.72

    ax.scatter(x_train, y_train, s=50, color=col[0] , edgecolors=col[1], label=labels[0])  
    ax.scatter(x_test, y_test, s=50, color=col[2], edgecolors=col[3], label=labels[1])  
    
    ax.set_xlim(-ll, ll)
    ax.set_ylim(-ll, ll)
    ax.set_xticks([-0.5,-0.25,0,0.25,0.5,])
    if kk==0: 
        ax.set_yticks([-0.5,-0.25,0,0.25,0.5,])
    else:
        ax.set_yticks([])
    ax.plot(np.arange(-ll,ll,0.01),np.arange(-ll,ll,0.01),  'k--',alpha=0.7)  
    ax.axhspan(-ll, 0, .5, 0 ,color='#FFE500', alpha=0.2)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')
    
    ax.set_title(title,x=0.76,y=.02,backgroundcolor='w', color='k')
    ax.set_xlabel("True $\langle \mathcal{W} \\rangle$")
    if kk==0: ax.set_ylabel("Predicted $\langle \mathcal{W} \\rangle$")
    
    ax.legend(loc="upper left", frameon=False,handletextpad=0.0)
    ax.set_aspect('equal')
plt.show()




def cm_plot(cm, ax, title, cmap="Blues"):
    # Plot the heatmap manually
    #fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(cm, cmap=cmap, aspect="auto")

    # Add colorbar
    #cbar = fig.colorbar(cax)
    #cbar.ax.set_ylabel('Percentage')

    # Set axis labels and ticks
    
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$\langle \mathcal{W} \\rangle<0$", "$\langle \mathcal{W} \\rangle>0$"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["$\langle \mathcal{W} \\rangle>0$", "$\langle \mathcal{W} \\rangle<0$"],rotation=90, va="center")
    ax.set_title(title)

    # Get color limits to determine contrast
    min_val, max_val = cm.min(), cm.max()
    threshold = (min_val + max_val) / 2  # Midpoint for color contrast

    # Add annotations with percentage values and adaptive text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > threshold else "black"  # Adaptive text color
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
    
    ax.set_aspect('equal')
plt.show()

def pauli_plot(title,x_train,y_train,x_test,y_test,ax,kk,labels,col):
    #'#9CEBA8','#5ABE69','#148125','#9CEBA8'
    ax.scatter(x_train, y_train, s=50, color=col[0] , edgecolors=col[1], label=labels[0])  
    ax.scatter(x_test, y_test, s=50, color=col[2], edgecolors=col[3], label=labels[1]) 
    
    ll=1.33
    ax.set_xlim(-ll, ll)
    ax.set_ylim(-ll, ll)
    
    ax.set_title(title,x=0.22,y=.8,backgroundcolor='w', color='k')

    ax.axhline(1.0, color='gray', linewidth=1.0, linestyle='--')
    ax.axhline(0.5, color='gray', linewidth=1.0, linestyle='--')
    ax.axhline(0, color='gray', linewidth=1.0, linestyle='--')
    ax.axhline(-0.5, color='gray', linewidth=1.0, linestyle='--')
    ax.axhline(-1.0, color='gray', linewidth=1.0, linestyle='--')

    ax.axvline(1.0, color='gray',   linewidth=0.5, linestyle='--')
    ax.axvline(0.5, color='gray',   linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray',     linewidth=0.5, linestyle='--')
    ax.axvline(-0.5, color='gray',  linewidth=0.5, linestyle='--')
    ax.axvline(-1.0, color='gray',  linewidth=0.5, linestyle='--')

    ax.plot(np.arange(-ll,ll,0.1),np.arange(-ll,ll,0.1), 'k--')  

    if kk>=12: 
        ax.set_xlabel("True Value")
    if np.mod(kk,4)==0: 
        ax.set_ylabel("Predicted Value")
    if np.mod(kk,4)!=0:
        ax.set_yticks([])
    if np.mod(kk,4)==0:
        ax.set_yticks([-1,-0.5,0,0.5,1])
    if kk<=11: 
        ax.set_xticks([])
    if kk>=12:
        ax.set_xticks([-1,-0.5,0,0.5,1])
    
    #ax.legend(loc="lower right", frameon=False)
    ax.set_aspect('equal')
plt.show()