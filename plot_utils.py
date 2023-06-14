import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import matplotlib.figure as mfig
import os
import numpy as np

ORIGIN = {
        # tick_params width와 line_width는 같도록 설정하자.
        'custom_font_dict' : {'fontname': 'Arial',
                              'fontsize' : 34,
                              'fontweight' : 'bold'},
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.tick_params.html
        'tick_params' : {'direction' : 'in',
                         'pad' : 10,
                         'width' : 10, # ticker 너비
                         'length' : 10 # ticker 높이
                         },
        'line_width' : 10,
        'plot_color' : 'k',
        'plot_linewidth' : 4,
        'other_color' : 'r',
        'figsize' : (15,11)
    }

def add_line(ax : plt.Axes , fig : mfig.Figure, idx : int, direction, **kwargs) :
    '''
    특정 index에 대해서 보조선을 그려주는 함수
    
    ax : plt.Axes, ex) ax = fig.add_subplot(111)\n
    fig : matplotlib.figure.Figure, ex) fig = plt.figure()\n
    idx : (int,list) 데이터의 몇 번째 값에 대해서 보조선을 그릴지\n
    direction : [both, horizontal, vertical], ex) horizontal : x축과 평행한 선, vertical : y축과 평행한 선, both : 둘다\n
    kwargs = {'linestyle':str, 보조선의 linestyle (plt의 linestyle과 동일) ex) '--'\n
              'linewidth' : float, 보조선의 두께\n
              'linecolor' : str, 보조선의 색깔 (plt의 color와 동일)\n
              'showvalue' : bool, 해당 index의 값을 표시할 지 (True or False)\n
              'prefix' : (str,list), 해당 index 값 앞에 표시할 내용 ex) value = 0.2, prefix = 'pre' -> result : 'pre0.2'\n
              'postfix' : (str,list), 해당 index 값 뒤에 표시할 내용 ex) value = 0.2, postfix = 'post' -> result : '0.2post'\n
              'precision' : (int,list), 해당 index 값을 소수점 몇 째 자리까지 표시할 지\n
              'x_padding' : float, 해당 index의 값을 x 방향으로 얼마만큼 띄워서 표시할 지 -1에서 1사이의 값을 넣을 수 있다.\n
              'y_padding' : float, 해당 index의 값을 y 방향으로 얼마만큼 띄워서 표시할 지 -1에서 1사이의 값을 넣을 수 있다.\n
              }
    ***********************************************************************************\n
    list type도 지원되는 키워드의 경우는 하나의 figure에 여러 plot을 그리는 상황을 지원한다.\n
    ax.plot(plot1)\n
    ax.plot(plot2)\n
    다음과 같은 상황에 대해서 순서대로 list에 값을 넣는다.\n
    ex. idx = [50,100] : plot1의 50번째 idx에 대해, plot2의 100번째 idx에 대해 보조선을 긋는다.\n
    ex. postfix = ['','%'] plot1의 value에 대해서는 아무것도 붙히지 않고, plot2의 value에 대해서는 뒤에 %를 붙힌다.\n
    '''
    
    # Initialize
    linestyle = plt.rcParams['lines.linestyle']
    linewidth = plt.rcParams['lines.linewidth']
    linecolor = plt.rcParams['lines.color']
    showvalue = False
    textcolor = plt.rcParams['text.color']
    prefixes = ""
    postfixes = ""
    precision = 1
    x_padding = 0
    y_padding = 0
    
    # 키워드 값이 존재하면 그 값으로 대체
    direction_option = ['both','horizontal','vertical']
    if direction not in direction_option :
        ValueError(f"Invalid option. Valid options are: {', '.join(direction_option)}")
    if 'linestyle' in kwargs :
        linestyle = kwargs['linestyle']
    if 'linewidth' in kwargs :
        linewidth = kwargs['linewidth']
    if 'linecolor' in kwargs :
        linecolor = kwargs['linecolor']
    if 'showvalue' in kwargs :
        showvalue = kwargs['showvalue']
    if 'textcolor' in kwargs :
        textcolor = kwargs['textcolor']
        showvalue = True
    if 'prefix' in kwargs :
        prefixes = kwargs['prefix']
        showvalue = True
    if 'postfix' in kwargs :
        postfixes = kwargs['postfix']
        showvalue = True
    if 'precision' in kwargs :
        precision = kwargs['precision']
        showvalue = True
    if 'x_padding' in kwargs :
        x_padding = kwargs['x_padding']
        assert -1<=x_padding<=1, "Invalid value : 0<=padding<=1"
        # padding을 그냥 숫자로 하면 화면 밖으로 벗어날 수 있다. 따라서 하나의 tick의 크기에 대한 비율로 정의하였다.
        x_major_locator = ax.xaxis.get_major_locator()
        x_major_ticklocs = x_major_locator.tick_values(ax.get_xlim()[0], ax.get_xlim()[1])
        x_major_spacing = x_major_ticklocs[1] - x_major_ticklocs[0]
        x_padding = x_major_spacing * x_padding
        showvalue = True
    if 'y_padding' in kwargs :
        y_padding = kwargs['y_padding']
        assert -1<=y_padding<=1, "Invalid value : 0<=padding<=1"
        
        y_major_locator = ax.yaxis.get_major_locator()
        y_major_ticklocs = y_major_locator.tick_values(ax.get_ylim()[0], ax.get_ylim()[1])
        y_major_spacing = y_major_ticklocs[1] - y_major_ticklocs[0]
        y_padding = y_major_spacing * y_padding
        showvalue = True    
    
    # plot이 여러개 있을 때를 상정하여 일반화하여 코드를 짰기 때문에 plot이 하나일 때도 함수 내부에서 list화 하여 표현
    idxs = idx
    if isinstance(idx, int) :
        idxs = [idx]
    if isinstance(prefixes, str) :
        prefixes = [prefixes]
    if isinstance(postfixes, str) :
        postfixes = [postfixes]
        
    # plot 상에 그려지는 값의 범위 (정의역, 공역의 범위)
    min_x, max_x = ax.get_xbound()
    min_y, max_y = ax.get_ybound()
    
    for i,line in enumerate(ax.get_lines()) : # 각 plot에 대해서
        idx = idxs[i]
        # 혹시나 그래프의 수보다 prefix, postfix의 값이 적을 때를 대비하여 ''값을 넣어준다.
        prefix = prefixes[i] if len(prefixes) > i else ''
        postfix = postfixes[i] if len(postfixes) > i else ''
        
        data = line.get_xydata() # [x y] pairs
        # 특정 y 값까지 dashline 그리기
        # data[idx][0] : x value, data[idx][1] : y value
        if direction == 'both' or direction == 'horizontal' :
            ax.axhline(data[idx][1], xmax = (idx - min_x)/(max_x - min_x), linestyle=linestyle, linewidth = linewidth, color = linecolor, label=f'horizontal line{i}')
        if direction == 'both' or direction == 'vertical' :
            ax.axvline(data[idx][0], ymax = (data[idx][1] - min_y)/(max_y - min_y), linestyle=linestyle, linewidth = linewidth, color = linecolor, label=f'vertical line{i}')
        # 값 표시하기
        if showvalue :
            x_screen, y_screen = ax.get_xaxis().get_transform().transform(idx), ax.get_yaxis().get_transform().transform(data[idx][1])
            ax.text(x_screen - x_padding, y_screen - y_padding, f"{prefix}{data[idx][1]:.{precision}f}{postfix}", ha='right', va='top', color = textcolor)
    
    return ax, fig

def change_style(ax : plt.Axes, fig : mfig.Figure , **kwargs) :
    '''
    plot의 스타일을 바꿔주는 함수
    
    ax : plt.Axes, ex) ax = fig.add_subplot(111)\n
    fig : matplotlib.figure.Figure, ex) fig = plt.figure()\n
    kwargs = {'style' : str, ["origin"] 중 하나. 이미 설정해놓은 스타일로 그래프를 바꾼다.\n
              'custom_font_dict' : dict(), plot에 나타나는 모든 글자들의 font 정보를 바꾼다. ex) {'fontname': 'Arial','fontsize' : 34,'fontweight' : 'bold'}\n
              'tick_params' : dict(), 축의 tick(값을 표시하는 수직선) 정보를 변경한다. \n
                                                                                    ex) {'direction' : 'in', # ticker 방향 ['in','out']\n
                                                                                        'pad' : 10, # 축과 ticker 간의 간격\n
                                                                                        'width' : 10, # ticker 너비\n
                                                                                        'length' : 10 # ticker 높이\n
                                                                                        }\n
              'linewidth' : float, figure 테두리선의 두께 변경\n
              'plot_linewidth' : float, plot 선의 두께 변경\n
              'plot_color' : str, plot 선의 색깔 변경, plt color 값을 따른다. ex) 'r' : red, 'k' : black, 'b' : blue\n
              'title' : str, title에 어떤 값을 넣을지\n
              'xlabel' : str, xlabel에 어떤 값을 넣을지\n
              'xticklabel_mapping' : ftn, x 값에 대해서 함수를 취해준다. ex) xticklabel_mapping : int  | [0.0 1.0 2.0] --int--> [0 1 2]\n
              'ylabel' : str, ylabel에 어떤 값을 넣을지\n
              'yticklabel_mapping' : ftn, y 값에 대해서 함수를 취해준다. ex) yticklabel_mapping : int  | [0.0 1.0 2.0] --int--> [0 1 2]\n
              'use_math_text' : bool, 수학적 표기방법을 사용할지, ex) True : 소수점 값들을 10^-5와 같은 형식으로 바꿔준다.\n
              'save' : bool, figure를 저장할지 말지\n
              'filename' : str, figure를 저장할 때 파일 이름\n
              'filepath' : str, figure 저장 경로\n
              'filetype' : str, figure 저장 타입 ex) 'png', 'svg'\n
              'transparent' : bool, figure의 뒷 배경 색을 투명하게 할지 말지\n
              'dpi' : int, figure의 해상도\n
              }
    '''
    # Initialize
    figsize = plt.rcParams['figure.figsize']
    default_font = fm.FontProperties()
    custom_font_dict = {'fontname': default_font.get_name(), 'fontsize' : default_font.get_size(), 'fontweight' : default_font.get_weight()}
    linewidth = plt.rcParams['axes.linewidth']
    plot_color = plt.rcParams['lines.color']
    plot_linewidth = plt.rcParams['lines.linewidth']
    tick_params = None
    xticklabel_mapping = lambda x : x
    yticklabel_mapping = lambda x : x
    use_math_text = False
    save = False
    path = ''
    filetype = 'png'
    filename = 'figure'
    transparent = True
    dpi = 300
    # 키워드 값이 존재하면 그 값으로 대체
    if 'style' in kwargs:
        style = globals()[kwargs['style'].upper()]
        custom_font_dict = style['custom_font_dict']
        tick_params = style['tick_params']
        linewidth = style['line_width']
        plot_color = style['plot_color']
        plot_linewidth = style['plot_linewidth']
        figsize = style['figsize']
    if 'custom_font_dict' in kwargs :
        custom_font_dict = kwargs['custom_font_dict']
    if 'tick_params' in kwargs :
        tick_params = kwargs['tick_params']
    if 'linewidth' in kwargs :
        linewidth = kwargs['linewidth']
    if 'plot_linewidth' in kwargs :
        plot_linewidth = kwargs['plot_linewidth']
    if 'plot_color' in kwargs :
        plot_color = kwargs['plot_color']
    if 'xticklabel_mapping' in kwargs :
        xticklabel_mapping = kwargs['xticklabel_mapping']
    if 'yticklabel_mapping' in kwargs :
        yticklabel_mapping = kwargs['yticklabel_mapping']
    if 'use_math_text' in kwargs :
        use_math_text = kwargs['use_math_text']
    if 'save' in kwargs :
        save = kwargs['save']
    if 'path' in kwargs :
        path = kwargs['filepath']
        save = True
    if 'filetype' in kwargs :
        filetype = kwargs['filetype']
        save = True
    if 'filename' in kwargs :
        filename = kwargs['filename']
        save = True
    if 'transparent' in kwargs :
        transparent = kwargs['transparent']
        save = True
    if 'dpi' in kwargs :
        dpi = kwargs['dpi']
        save = True
    
    
    
    
    
        
    ############ custom figsize ############
    # 사용자가 figsize를 따로 설정하지 않았고 style에서 설정된 figsize가 있다면 그 figsize로 변경된다.
    if all(fig.get_size_inches() == np.array(plt.rcParams['figure.figsize'])) :
        fig.set_size_inches(*figsize)
    # 사용자가 따로 figsize를 설정했을 때, 그 값을 기준으로 style값이 scaling 되어 적용된다.
    else :
        if 'style' in kwargs :
            scaling = min(fig.get_size_inches())/min(globals()[kwargs['style'].upper()]["figsize"])
            for k,v in custom_font_dict.items() :
                if isinstance(v,int) or isinstance(v,float) :
                    custom_font_dict[k] = v * scaling
            for k,v in tick_params.items() :
                if isinstance(v,int) or isinstance(v,float) :
                    tick_params[k] = v * scaling
            linewidth = linewidth * scaling
            plot_linewidth = plot_linewidth * scaling
            
    
    ############ custom font ############
    # label 설정
    if 'xlabel' in kwargs :
        ax.set_xlabel(kwargs['xlabel'], fontdict=custom_font_dict)
    if 'ylabel' in kwargs :
        ax.set_ylabel(kwargs['ylabel'], fontdict=custom_font_dict)
    if 'title' in kwargs :
        ax.set_title(kwargs['title'], fontdict=custom_font_dict)
    # ticklabel 값을 mapping 함수에 기반하여 바꾼뒤, font를 적용한다.
    # mapping 함수를 설정하지 않았다면, lambda x : x에 의해 아무런 값도 변경되지 않는다.
    ax.set_xticklabels(list(map(xticklabel_mapping, ax.get_xticks())), **custom_font_dict)
    ax.set_yticklabels(list(map(yticklabel_mapping, ax.get_yticks())), **custom_font_dict)
    
    # 소수점이 아래값이 많은 애들은 math_text를 쓰지 않으면 font 변경시 e-5와 같은 것들이 xtick에 전부 표시된다.
    if use_math_text :
        formatter = ticker.ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.get_offset_text().set_fontweight(custom_font_dict['fontweight'])
        ax.xaxis.get_offset_text().set_fontsize(custom_font_dict['fontsize'])
        ax.xaxis.get_offset_text().set_fontname(custom_font_dict['fontname'])
        
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontweight(custom_font_dict['fontweight'])
        ax.yaxis.get_offset_text().set_fontsize(custom_font_dict['fontsize'])
        ax.yaxis.get_offset_text().set_fontname(custom_font_dict['fontname'])
    # 모든 text에 대해서 font 적용
    for text in ax.texts :
        text.set(**custom_font_dict)
    # custom tick_params : tick 값에 대해서 font 적용
    if tick_params is not None :
        ax.tick_params(**tick_params)
    #####################################
    
    ############ plot color ############
    for line in ax.get_lines() :
        # see def add_line(), ax.hline or ax.vline's color and width is already defined by add_line()
        # 보조선은 이미 색깔이 지정되므로 따로 설정하지 않는다.
        if "horizontal line" in line.get_label() or "vertical line" in line.get_label()  :
            continue
        # plot의 색깔을 바꾼다.
        line.set_color(plot_color)
        line.set_linewidth(plot_linewidth)
    #####################################
    
    # custom linewidth of axes
    # figure의 바깥선들의 두께를 바꾼다.
    [x.set_linewidth(linewidth) for x in ax.spines.values()]
    if save :
        fig.savefig(os.path.join(path,f"{filename}.{filetype}"), dpi=dpi, transparent=transparent)
    plt.show()
    
def generate_example(key='plot1') :
    '''
    예제 함수 만들어주는 함수
    key 값으로 plot1, plot2, hist가 있다.
    '''
    if key == 'plot1' :
        value = [2.9, 1.56, 1.52, 1.03, 1.16, 0.59, 1.07, 0.56, 0.51, 0.38, 0.63, 0.8, 0.3, 0.42, 0.48, 
                 0.37, 0.29, 0.33, 0.4, 0.35, 0.39, 0.46, 0.29, 0.21, 0.24, 0.3, 0.24, 0.42, 0.11, 0.26, 
                 0.2, 0.31, 0.29, 0.33, 0.42, 0.2, 0.34, 0.26, 0.26]
    elif key == 'plot2' :
        value = [26.0, 47.26, 61.4, 69.51, 74.56, 77.86, 80.11, 81.85, 83.09, 84.12, 85.02, 85.59, 86.19, 
                 86.74, 87.16, 87.45, 87.86, 88.08, 88.39, 88.56, 88.85, 89.06, 89.15, 89.29, 89.37, 89.45, 
                 89.6, 89.76, 89.76, 89.86, 89.92, 90.04, 90.02, 90.26, 90.24, 90.2, 90.3, 90.36, 90.39]
    elif key == 'hist' :
        rng = np.random.RandomState(10)
        value = np.hstack((rng.normal(size=1000),
                           rng.normal(loc=5,scale=2, size=1000)))
    return value
