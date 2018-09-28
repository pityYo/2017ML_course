#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
using namespace std;
using namespace Eigen;


int Attr_num = 9; // fixed
int sort_flag = 0;
int KNN_size = 1;

double convert_double(const string &str){
	stringstream ss(str);
	double ans;
	ss >> ans;
	return ans;
}
int convert_int(const string &str){
	stringstream ss(str);
	int ans;
	ss >> ans;
	return ans;
}

class Data{
public:
	friend class Node;
public:
	Data(){}
	Data(Data a, vector<double>tmp){
		name = a.name;
		type = a.type;
		index = a.index;
		for(int i = 0;i < tmp.size(); ++i){
			attr[i] = tmp[i];
		}
	}
	Data(const string &str){
		string tmp;
		istringstream line_in(str); 
		getline(line_in, tmp, ','); index = convert_int(tmp);
		getline(line_in, tmp, ','); name = tmp; // name
		for(int i = 0;i < Attr_num; ++i){
			getline(line_in, tmp, ',');
			attr[i] = convert_double(tmp);
		}
		getline(line_in, tmp, ','); type = tmp;
	}
	bool operator < (Data& b){ return this->attr[sort_flag] < b.attr[sort_flag]; }
	bool operator ==(Data& b){ return this->attr[sort_flag] ==b.attr[sort_flag]; }
	double operator[](const int &idx){
		/*if(idx >= Attr_num) {
			cout <<"over_flow\n";
			exit(-1);
		}*/
		return attr[idx];
	}
	string get_type(){
		return type;
	}
	int get_index(){
		return index;
	}
	
private:
	string name, type;
	double attr[9];
	int index;
};

class Node{
public:
	friend bool operator < (Node, Node);
public:
	Node(){}
	Node(const Data &input, const int idx): 
		data(input), attr_flag(idx){
			l_idx = r_idx = -1;
		}
	bool operator < (Data &data_cmp) { return data[attr_flag] < data_cmp[attr_flag]; }
	bool operator >=(Data &data_cmp) { return data[attr_flag] >=data_cmp[attr_flag]; }
	void set_child(int a, int b){
		l_idx = a;
		r_idx = b;
	}
	int left_child()  { return l_idx;}
	int right_child() { return r_idx;}
	double distance(Data &a){
		double ans = 0, x;
		for(int i = 0;i < Attr_num; ++i){
			x = data.attr[i] - a.attr[i];
			ans += x * x;
		}
		return sqrt(ans);
	}
	double project(Data &a){
		return abs(a.attr[attr_flag] - data.attr[attr_flag]);
	}
	string get_type(){
		return data.get_type();
	}
	int get_index(){
		return data.index;
	}
private:
	Data data;
	int attr_flag;
	int l_idx, r_idx;
};

Data test;
class KD_tree{
public:
	friend string vote(KD_tree*,int);
	friend void store_info(KD_tree*,vector<int>*);
public:
	KD_tree(){}
	void set_root(const int &a){
		root = a;
	}
	int build_KD_tree(vector<Data>& train_data, const int& idx){
		sort_flag = idx;
		int mid = train_data.size()/2;
		nth_element(train_data.begin(), 
					train_data.begin() + mid, 
					train_data.end());
		vector<Data>L(train_data.begin(), train_data.begin() + mid);
		vector<Data>R(train_data.begin()+mid+1, train_data.end());
		
		node.push_back(Node(train_data[mid], sort_flag));
		int ans = node.size() - 1, l_idx = -1, r_idx = -1;
		if(L.size() > 0){
			l_idx = build_KD_tree(L, (idx+1)%Attr_num);
		}
		if(R.size() > 0){
			r_idx = build_KD_tree(R, (idx+1)%Attr_num);
		}
			
		node[ans].set_child(l_idx, r_idx);
		return ans;
	}
	
	void KNN(int idx){
	//now only for nearest	
		Node tmp, ans = node[idx];
		int nxt, opp;
		if(node[idx] >= test){
			nxt = node[idx].left_child();
			opp = node[idx].right_child();
		}
		else {
			nxt = node[idx].right_child();
			opp = node[idx].left_child();
		}
		if(nxt != -1) KNN(nxt);
		n_nearest.push(node[idx]);
		if(n_nearest.size() < KNN_size && opp != -1) KNN(opp);
		else if(n_nearest.size() >= KNN_size){
			while(n_nearest.size() > KNN_size){
				n_nearest.pop();
			}
			tmp = n_nearest.top();
			if(tmp.distance(test) > node[idx].project(test) && opp != -1)
				KNN(opp);
				
		}
		
	}
	void clean_priority_queue(){
		while(!n_nearest.empty()){
			n_nearest.pop();
		}
	}
	
private:
	vector<Node>node;
	priority_queue<Node>n_nearest;
	int root;
};


bool operator < (Node a,Node b){
	//farthest to nearest
	return a.distance(test) < b.distance(test); 
}
void store_info(KD_tree* tree,vector<int> *ptr){
	priority_queue<Node> pq = tree->n_nearest;
	while(!pq.empty()){
		Node tmp = pq.top();
		ptr->push_back(tmp.get_index());
		pq.pop();
	}
}
string vote(KD_tree* itr, int id){
	vector<string>s;
	map<string, int>ap;
	vector<Node>foo;
	while(!(itr->n_nearest).empty()){
		Node tmp = (itr->n_nearest).top();
		foo.push_back(tmp);
		(itr->n_nearest).pop();
		string type = tmp.get_type();
		if(!ap[type]) s.push_back(type);
		ap[type]++;
	}
	string ans;
	int jud = -1;
	for(int i = 0;i < s.size(); ++i){
		if(jud < ap[s[i]]){
			jud = ap[s[i]];
			ans = s[i];
		}
	}
	
	return ans;
}
 
void computeCov(MatrixXd &X, MatrixXd &C)  
{  
    //计算协方差矩阵C = XTX / n-1;  
	MatrixXd centered = X.rowwise() - X.colwise().mean();
	C = centered.adjoint()*centered;
	
	
  //  C = X.adjoint() * X;  
    //C = C.array() / X.rows() - 1;  
}  
void computeEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val)  
{  
    //计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列  
    SelfAdjointEigenSolver<MatrixXd> eig(C);
  
    vec = eig.eigenvectors();  
    val = eig.eigenvalues();  
}  
int computeDim(MatrixXd &val)  
{  
    int dim;  
    double sum = 0;  
    for (int i = val.rows()-1; i >= 0; --i)  
    {  
        sum += val(i, 0);  
        dim = i;  
          
        if (sum / val.sum() >= 0.95)  
            break;  
    }  
    return val.rows() - dim;  
} 

int main(int argc, char **argv){
	string input, test_data;
	if(argc >= 3){
		//KNN_size = atoi(argv[1]);
		input = argv[1];
		test_data = argv[2];
	}
	else{
		cout <<"input training data file name and test data file name\n";
		exit(-1);
	}
	ifstream fin;
	fin.open(input);
	if(!fin){
		cout <<"opening file error\n";
		exit(-1);
	}
	vector<Data>con;
	vector<Data>haha;
	string str, tmp;
	//int cnt = 0;
	getline(fin, str, '\n'); // useless input
	while(getline(fin, str, '\n')) {
		//cout << str << endl;
		con.push_back(Data(str));
	}

	KD_tree tree;
	//int correct = 0;
	int root = tree.build_KD_tree(con, 0);
	tree.set_root(root);
	fin.close();
	fin.open(test_data);
	if(!fin){
		cout <<"test opened failed\n";
		exit(-1);
	}
	getline(fin,str,'\n');
	while(getline(fin,str)){
		//cout << str << endl;
		haha.push_back(Data(str));
	}
	int K[4] = {1,5,10,100};
	int correct[4] = {};
	vector<int>ans_idx[4];
	for(int i = 0;i < 4; ++i){
		KNN_size = K[i];
		ans_idx[i].clear();
		correct[i] = 0;
		for(int j = 0;j < haha.size(); ++j){
			test = haha[j];
			tree.clean_priority_queue();
			tree.KNN(0);
			if(j < 3) store_info(&tree,&ans_idx[i]);
			string guess = vote(&tree, j);
			if(guess == test.get_type())
				correct[i]++;
		}
		cout << "KNN accuracy : " << double(correct[i]) / double(haha.size()) << endl;
		for(int j = 0;j < 3; ++j){
			for(int k = KNN_size - 1;k >= 0; --k){
				//if(k) cout << " ";
				cout << ans_idx[i].at(j*KNN_size + k)<<" ";
			}
			cout << endl;
		}
		cout <<"\n";
	}
	//PCA
	const int Xm = con.size();
	MatrixXd X(Xm,9), C(9,9);
	MatrixXd vec, val;
	for(int i = 0;i < con.size(); ++i){
		for(int j = 0;j < 9;++j){
			X(i,j) = con[i][j];
		}
	}
	computeCov(X,C);
	//cout <<"C :\n";
	//cout << C << endl;
	computeEig(C,vec,val);
	//cout << "vec:\n";
	//cout << vec<<endl;
	//cout <<"val :\n";
	//cout << val << endl;
	int dim = computeDim(val);
	MatrixXd res = X * vec.rightCols(dim);
	//cout << res << endl;
	vector<Data>pca_train;
	Attr_num = dim;
	for(int i = 0;i < con.size(); ++i){
		vector<double>tmp;
		for(int j = 0;j < dim; ++j)
			tmp.push_back(res(i,j));
		pca_train.push_back(Data(con[i],tmp));
	}
	KD_tree pca_tree;
	pca_tree.build_KD_tree(pca_train,0);
	int pca_correct = 0;
	KNN_size = 5;
	for(int i = 0;i < haha.size(); ++i){
		Data td = haha[i];
		MatrixXd sig(1,9), tm;
		for(int j = 0;j < 9; ++j){
			sig(0,j) = td[j];
		}
		tm = sig * vec.rightCols(dim);
		vector<double>tmp;
		for(int j = 0;j < dim; ++j)
			tmp.push_back(tm(0,j));
		test = Data(haha[i],tmp);
		pca_tree.clean_priority_queue();
		pca_tree.KNN(0);
		string guess = vote(&pca_tree, 0);
		if(guess == test.get_type())
			pca_correct++;
	}
	cout <<"K = " << KNN_size <<", KNN_PCA accuracy: " << double(pca_correct)/haha.size() << endl;
	
	
	fin.close();
	//cout << "accuracy : " << double(correct)/36.0 << endl;
	return 0;
}
