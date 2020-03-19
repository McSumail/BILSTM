from sklearn.utils import shuffle
from enum import Enum
import itertools
import pickle
from gensim.models.keyedvectors import KeyedVectors
import jieba
import jieba.posseg as pseg

class FourClass(Enum):
    none=0
    ea=1
    expe=2
    expa=3

def Read_Hotel():
    print("data processing begin")
    data={}
    x,y=[],[]
    with open("hotel.txt", "r", encoding="utf-8") as f:
        line=f.readlines()
        i=3
        while i<len(line):
            # print(i)
            content=line[i]
            att,eva,exp,ea,expea=[],[],[],[],[]
            j=0
            while True:
                if j>=len(content): break
                if content[j]=='<':
                    if content[j+1:j+4]=='exp':
                        beg=j+5
                        end=content.find('>',beg)
                        expea.append(content[beg:end])
                        beg=content.find('>',j+4)+1
                        end=content.find('<',beg)
                        exp.append(content[beg:end])
                        j=end
                        continue
                    elif content[j+1:j+2]=='e':
                        beg=j+2
                        end=content.find('>',beg)
                        ea.append(content[beg:end])
                        beg=content.find('>',j+2)+1
                        end=content.find('<',beg)
                        eva.append(content[beg:end])
                        j=end
                        continue
                    elif content[j+1:j+2]=='a':
                        beg=content.find('>',j+2)+1
                        end=content.find('<',beg)
                        att.append(content[beg:end])
                        j=end
                        continue
                j+=1
            i+=6
            tagea=[]
            iterea1=0
            while iterea1<len(ea):
                iterea2=0
                while iterea2<len(ea[iterea1]):
                    if ea[iterea1][iterea2]=='a':
                        if ea[iterea1].find('-',iterea2)>0:
                            tagea.append((int(ea[iterea1][iterea2+1:ea[iterea1].find('-',iterea2)]),iterea1))
                        else:tagea.append((int(ea[iterea1][iterea2+1:]),iterea1))
                    iterea2+=1
                iterea1+=1
            tagexpa=[]
            tagexpe=[]
            iterexpea1=0
            while iterexpea1<len(expea):
                iterexpea2=3
                while iterexpea2<len(expea[iterexpea1]):
                    if expea[iterexpea1][iterexpea2]=='a':
                        if expea[iterexpea1].find('-',iterexpea2)>0:
                            tagexpa.append((int(expea[iterexpea1][iterexpea2+1:expea[iterexpea1].find('-',iterexpea2)]),iterexpea1))
                        else:tagexpa.append((int(expea[iterexpea1][iterexpea2+1:]),iterexpea1))
                    if expea[iterexpea1][iterexpea2]=='e':
                        if expea[iterexpea1].find('-',iterexpea2)>0:
                            tagexpe.append((int(expea[iterexpea1][iterexpea2+1:expea[iterexpea1].find('-',iterexpea2)]),iterexpea1))
                        else:tagexpe.append((int(expea[iterexpea1][iterexpea2+1:]),iterexpea1))
                    iterexpea2+=1
                iterexpea1+=1
            iter1=0
            while iter1<len(att):
                iter2=0
                while iter2<len(eva):
                    #
                    words = pseg.cut(att[iter1]+eva[iter2])
                    result = []
                    for w in words:
                        result.append(w.word)
                    x.append(result)
                    flag=False
                    for couple in tagea:
                        if iter1==couple[0] and iter2==couple[1]:
                            flag=True
                            break
                    if flag==True : y.append(1)
                    else: y.append(0)
                    iter2+=1
                iter1+=1
            iter1=0
            while iter1<len(eva):
                iter2=0
                while iter2<len(exp):
                    #
                    words = pseg.cut(eva[iter1]+exp[iter2])
                    result = []
                    for w in words:
                        result.append(w.word)
                    x.append(result)
                    flag=False
                    for couple in tagexpe:
                        if iter1==couple[0] and iter2==couple[1]:
                            flag=True
                            break
                    if flag==True : y.append(2)
                    else: y.append(0)
                    iter2+=1
                iter1+=1
            iter1=0
            while iter1<len(att):
                iter2=0
                while iter2<len(exp):
                    #
                    words = pseg.cut(att[iter1]+exp[iter2])
                    result = []
                    for w in words:
                        result.append(w.word)
                    x.append(result)
                    flag=False
                    for couple in tagexpa:
                        if iter1==couple[0] and iter2==couple[1]:
                            flag=True
                            break
                    if flag==True : y.append(3)
                    else: y.append(0)
                    iter2+=1
                iter1+=1
    # for q in range(len(x)):
    #     if y[q]==2:
    #      print(x[q]+'   '+str(y[q]))
    # devide 把训练数据分为十组
    dev_idx = len(x) // 10
    data["train_x"], data["train_y"] = x[:dev_idx*7], y[:dev_idx*7]
    data["dev_x"], data["dev_y"] = x[dev_idx*7:dev_idx*9], y[dev_idx*7:dev_idx*9]
    data["test_x"],data["test_y"] = x[dev_idx*9:], y[dev_idx*9:]
    print("data processing finish")
    return data

def save_model(model, params):
    path = "saved_models/{model}_{epoch}.pkl".format(model=params['MODEL'],epoch=params['EPOCH'])
    #pickle.dump(obj, file, [,protocol])
    # 注释：序列化对象，将对象obj保存到文件file中去。
    # 参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），
    # protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。
    # 其中，1是老式的二进制协议；2是新二进制协议）。
    # file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，
    # 也可以是任何可以实现write()接口的对象。
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {p}!".format(p=path))

def load_model(params):
    path = "saved_models/{model}_{epoch}.pkl".format(model=params['MODEL'],epoch=params['EPOCH'])
    try:
        model = pickle.load(open(path, "rb"))
        print("Model in {p} loaded successfully!".format(p=path))

        return model
    except:
        print("No available model such as {p}.".format(p=path))
        exit()

# if __name__ == "__main__":
