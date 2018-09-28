#include <iostream>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <random>
#include <chrono>
using namespace std;

int sort_flag;
bool random_forest;
class Data{
public:
	Data(){}
	Data(double a, double b, double c, double d, string s = ""):
		sepal_length(a),
		sepal_width(b),
		petal_length(c),
		petal_width(d),
		catalog(s){}
	bool operator < (Data&b){
		return (*this)[sort_flag] < b[sort_flag];
	}
	double operator[](const string &str){
		if(str == "sepal_length") return sepal_length;
		else if(str == "sepal_width")  return sepal_width;
		else if(str == "petal_length") return petal_length; 
		else if(str == "petal_width")  return petal_width; 
		else return -1.0;
	
	}
	double operator [](const int& idx){
		switch(idx){
			case 0:  return sepal_length;
			case 1:  return sepal_width;
			case 2:  return petal_length;
			case 3:  return petal_width;
			default: return sepal_length;
		}
	}
	string operator ()(const int &idx){
		switch(idx){
			case 0:  return "sepal_length";
			case 1:  return "sepal_width";
			case 2:  return "petal_length";
			case 3:  return "petal_width";
			default: return "petal_width";
		}
	}
	string get_catalog(){ return catalog; }
	void show(){
		cout <<sepal_length<<"     "<<sepal_width<<"     "<<petal_length<<"     " <<petal_width<<"     " << catalog<<endl;;
	}
private:
	double sepal_length, sepal_width, petal_length, petal_width;
	string catalog;
};

class Node{
private:
	string feature, catalog;
	double judge;
	int r_idx, l_idx;
public:
	Node(){r_idx = l_idx = -1;}
	Node(string str = "", double d = 0.0):
		feature(str), judge(d) { r_idx = l_idx = -1; }
	bool operator < (Data &data) const { return judge < data[feature]; }
	bool operator > (Data &data) const { return judge > data[feature]; }
	bool operator ==(Data &data) const { return judge == data[feature];}
	bool is_leaf() { return (r_idx == -1 && l_idx == -1); }
	void set_child(const int &l, const int &r){ l_idx = l; r_idx = r;}
	int left_child()  { return l_idx;}
	int right_child() { return r_idx;}
	string get_feature() {return feature;}
	int no_choice() {
		if(l_idx == -1) return r_idx;
		else if(r_idx == -1) return l_idx;
		else return -1;
	}
};

class Decision_tree{
public:
	Decision_tree(){}
	~Decision_tree(){ node.clear(); }
	
	double get_maxIG(vector<Data>td, double *val){
		sort(td.begin(), td.end());
		
		double pre_sum[3][td.size()];
		string catalog;
		memset(pre_sum,0,sizeof(pre_sum));
		for(size_t i = 0; i < td.size(); ++i){
			catalog = td[i].get_catalog();
			pre_sum[0][i] = (i>0)? pre_sum[0][i-1]:0;
			pre_sum[1][i] = (i>0)? pre_sum[1][i-1]:0;
			pre_sum[2][i] = (i>0)? pre_sum[2][i-1]:0;
			if(catalog == "Iris-setosa")             pre_sum[0][i] += 1.0;
			else if(catalog == "Iris-versicolor")    pre_sum[1][i] += 1.0;
			else if(catalog == "Iris-virginica")     pre_sum[2][i] += 1.0;
			
		}
		
		vector<Data>::iterator itr;
		vector<int>::iterator s_itr;
		vector<int>split;
		string bef = "";
		for(itr = td.begin(); itr != td.end(); ++itr){
			catalog = itr->get_catalog();
			if(bef == "") bef = catalog;
			if(bef != catalog){
				split.push_back(itr-td.begin());
				bef = catalog;
			}
		}
		
		double max = -100000.0, tmp_IG;
		double org_entropy = 0.0;
		for(int i = 0;i < 3; ++i){
			double prob = pre_sum[i][td.size()-1]/(double)td.size();
			if(prob != 0.0)
				org_entropy += (-1.0) * (prob*log2(prob));
		}
		for(s_itr = split.begin(); s_itr != split.end(); ++s_itr){
			int idx = *s_itr;
			double entropy_1 = 0.0, entropy_2 = 0.0, prob;
			for(int i = 0;i < 3; ++i){
				prob = pre_sum[i][idx-1]/(double)(idx);
				if(prob != 0.0) 
					entropy_1 += (-1.0) * (prob*log2(prob));
			}
			for(int i = 0;i < 3; ++i){
				prob = (pre_sum[i][td.size()-1] - pre_sum[i][idx-1])/double(td.size()-idx);
				if(prob != 0.0)
					entropy_2 += (-1.0)*(prob * log2(prob));
			}
			double partition = double(idx)/double(td.size());
			double remainder = partition*entropy_1 + (1.0-partition)*entropy_2;
			tmp_IG = org_entropy - remainder;
			if(tmp_IG > max){
				max = tmp_IG;
				*val = (td[idx][sort_flag] +td[idx-1][sort_flag])/2;
				//cout <<"idx = " << idx <<" , threshold = " << *val<<endl;
			}
		}
		return max;
	}
	void choose(vector<Data> training_data, double *val, string *select){
		double IG = -10000.0, max = -100000.0;
		double t_threshold, threshold_in;
		int label = 0;
		for(int i = 0;i < 4 ; ++i){
			sort_flag = i;
			IG = get_maxIG(training_data, &threshold_in);
			
			if(IG > max){ 
				max = IG;
				label = i;
				t_threshold = threshold_in;
			}
		}
		
		*val = t_threshold;
		*select = training_data[0](label);
		
		return;
	}
	void build_tree(vector<Data>&training_data) {
		double threshold;
		string select;
		choose(training_data, &threshold, &select);
		
		node.push_back(Node(select, threshold));
		ID3(0, training_data);
	}
	bool all_same_catalog(vector<Data>d){
		vector<Data>::iterator itr;
		string s =d[0].get_catalog();
		for(itr = d.begin(); itr != d.end(); ++itr){
			
			if(s != itr->get_catalog()) return false;
		}
		return true;
	}
	void ID3(int idx, vector<Data>t_data){
		
		vector<Data>less, greater;
		less.clear();
		greater.clear();
		
		vector<Data>::iterator itr;
		
		for(itr = t_data.begin(); itr != t_data.end(); ++itr){
			if(node[idx] > (*itr)) less.push_back(*itr);
			else greater.push_back(*itr);
		}
		
		bool flag = (less.size() == 0 || greater.size() == 0);
		int l_idx = -1, r_idx = -1;
		if(flag){
			int vote[3] = {0,0,0};
			vector<Data>::iterator itr, end;
			if(less.size() == 0){		
				itr = greater.begin();
				end = greater.end();
				r_idx = node.size();
			}
			else{
				itr = less.begin();
				end = less.end();
				l_idx = node.size();
			}
			for(; itr != end; ++itr){
				string catalog = itr->get_catalog();
				if(catalog == "Iris-setosa")             ++vote[0];
				else if(catalog == "Iris-versicolor")    ++vote[1];
				else if(catalog == "Iris-virginica")     ++vote[2];
			}
			string feature = (vote[0] >= vote[1] && vote[0] >= vote[2])?"Iris-setosa":(
								(vote[1] >= vote[2])?"Iris-versicolor" : "Iris-virginica");
								
			node.push_back(Node(feature));
		}
		else{
			double threshold;
			string select;
			l_idx = node.size();
			if(all_same_catalog(less))
				node.push_back(Node(less[0].get_catalog()));
			else{
				choose(less, &threshold, &select);
				node.push_back(Node(select, threshold));
				ID3(l_idx, less);
			}
			r_idx = node.size();
			if(all_same_catalog(greater))
				node.push_back(Node(greater[0].get_catalog()));
			else{
				choose(greater, &threshold, &select);
				node.push_back(Node(select, threshold));
				ID3(r_idx, greater);
			}
		}
		node[idx].set_child(l_idx, r_idx);
	}
	string predict(int node_idx, Data &data){
		int nxt;
		if(node[node_idx].is_leaf()) return node[node_idx].get_feature();
		if((nxt = node[node_idx].no_choice()) != -1) return predict(nxt, data);
		else {
			if(node[node_idx] > data) nxt = node[node_idx].left_child();
			else nxt = node[node_idx].right_child();
			return predict(nxt, data);
		}
	}
private:
	//vector<Data>org_data;
	vector<Node>node;
};

double convert_double(string &s){
	stringstream ss(s);
	double ans;
	ss >> ans;
	return ans;
}
int convert(const string &s){
	if(s == "Iris-setosa") return 0;
	else if(s == "Iris-versicolor") return 1;
	else if(s == "Iris-virginica") return 2;
	else{
		cout <<"convert failed\n";
		exit(-1);
	}
}

vector<Decision_tree>forest;
vector<Data>tmp;
void K_fold_cross_validation(vector<Data>&org_data, bool random_forest){
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle (org_data.begin(), org_data.end(), std::default_random_engine(seed));
	
	int block_size = org_data.size()/5;
	
	double precision[3] = {0.0, 0.0, 0.0};
	double recall[3] = {0.0, 0.0, 0.0};
	double accuracy = 0.0;
	int s[3], g_t[3], g_f[3];
	for(int k = 0;k < 5; ++k){

		s[0] = s[1] = s[2] = 0;
		g_t[0] = g_t[1] = g_t[2] = 0;
		g_f[0] = g_f[1] = g_f[2] = 0;
		forest.clear();
		tmp.clear();
		if(random_forest){

			tmp.assign(org_data.begin(), org_data.end());
			tmp.erase(org_data.begin() + k*block_size, org_data.begin()+k*block_size+block_size);
			size_t training_size = tmp.size();
			for(int i = 0;i < 35; ++i){
				Decision_tree dt;
				shuffle (tmp.begin(), tmp.end(), std::default_random_engine(seed));
				vector<Data>input;
				input.assign(tmp.begin(), tmp.begin() + tmp.size() - 25);
				dt.build_tree(input);
				forest.push_back(dt);
			}
		}
		else{
			Decision_tree dt;
			vector<Data>input(org_data);
			int bound_l, bound_r;
			bound_l = k*block_size;
			bound_r = bound_l + block_size;
			input.erase(input.begin() + bound_l, input.begin() + bound_r);
			dt.build_tree(input);
			forest.push_back(dt);
		}
		vector<Data>test_data;
		vector<Data>::iterator itr;
		vector<Decision_tree>::iterator tree;
		int cata_idx, predict_idx, vote[3], result;
		string cata, cata_predict;
		test_data.assign(org_data.begin() + k*block_size, org_data.begin() + k*block_size + block_size);
		for(itr = test_data.begin(); itr != test_data.end(); ++itr){
			cata = itr->get_catalog();
			//cata_predict = tree.predict(0, *itr);
			cata_idx = convert(cata);
			vote[0] = vote[1] = vote[2] = 0;
			for(tree = forest.begin(); tree != forest.end(); ++tree){
				cata_predict = tree->predict(0, *itr);
				predict_idx = convert(cata_predict);
				vote[predict_idx]++;
			}
			result = (vote[0] >= vote[1] && vote[0] >= vote[2])?0:((vote[1] >= vote[2])?1:2);
			
			s[cata_idx]++;
			if(result == cata_idx) g_t[result]++;
			else g_f[result]++;
		}
		for(int i = 0; i < 3; ++i){
			precision[i] += double(g_t[i]) / double(g_t[i] + g_f[i]);
			recall[i] += double(g_t[i]) / double(s[i]);
		}
		accuracy += double(g_t[0] + g_t[1] + g_t[2]) / double(s[0] + s[1] + s[2]);
	}
	cout <<setprecision(3)<< accuracy/5.0 << endl;
	for(int i = 0;i < 3; ++i){
		cout <<setprecision(3)<< precision[i] / 5.0 <<" " << recall[i] / 5.0 << endl;
	}
	return;
}



int main(int argc, char* argv[]){
	ifstream fin("iris.txt");
	if(!fin){
		cout <<"open file failed\n";
		exit(-1);	
	}
	string input;
	random_forest = false;
	if(argc >= 3) {
		string t1(argv[1]), t2(argv[2]);
		if(t1 == "random" && t2 == "forest")
			random_forest = true;
	}
	//cout <<"hi\n";
	Decision_tree tree;
	vector<Data>org_data;
	while(getline(fin, input)){
		double tmp[4] = {-1, -1, -1, -1};
		int cnt = 0;
		string s = "";
		int len = input.length();
		//if(len <= 0) continue;
		for(int i = 0;i < len; ++i){
			if(input[i] == ','){
				tmp[cnt++] = convert_double(s);
				s = "";
			}
			else s += input[i];
		}
		if(tmp[0] == -1 || tmp[1] == -1 || tmp[2] == -1 || tmp[3] == -1)
			continue;
		org_data.push_back(Data(tmp[0], tmp[1], tmp[2], tmp[3], s));
	}
	
	K_fold_cross_validation(org_data, random_forest);
	
	
	return 0;
}
