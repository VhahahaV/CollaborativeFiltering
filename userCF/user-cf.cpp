//
// Created by CZQ on 2024/5/27.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <charconv>
#include <chrono>
#include <unordered_set>
//item nums : 3952
//user nums : 6040

using namespace std;

vector<string_view> split(const string_view &str, const char s){
    vector<string_view> result;
    int i = 0 , j = 0;
    for(;j < str.size(); j++){
        if(str[j] == s){
            result.emplace_back(str.substr(i,j-i));
            i = j+1;
        }
    }
    result.emplace_back(str.substr(i,j-i));
    return result;
}

template<typename T> using vec_t = std::vector<T>;
template<typename T> using mat_t = std::vector<std::vector<T>>;
template<typename T>
std::ostream& loadDisk(std::ostream& f, const mat_t<T>& scores){
    for(auto userScore: scores){
        for(size_t i=0; i<userScore.size(); i++){
            f << userScore[i];
            if(i+1!=userScore.size()){
                f << ',';
            } else{
                f << '\n';
            }
        }
    }
    return f;
}

class UserCF{
private:
    const int mNSimilar, mNRecommend;
    mat_t<int> mData;
    mat_t<int> mTrainSet, mTestSet;
    mat_t<double> mTrainSimilarMatrix;
    static const int gUserNum = 6040;
    static const int gItemNum =3952;
public:
    UserCF(const std::string& dataPath, int nSimilar, int nRecommend):
            mNSimilar(nSimilar), mNRecommend(nRecommend){
        std::ifstream f(dataPath, std::ios::in);
        if(!f.good()){
            throw std::runtime_error("open file '"+dataPath+"' error");
        }
        std::string line;
        int lastUserId = 1;
        int curUserId;
        auto parseLine = [&](const std::string& cur)->vec_t<int> {
            auto datItem = split(cur,':');
            vec_t<int> res;
            for(auto i : datItem){
                int num;
                if(i.empty())
                    continue;
                auto err = std::from_chars(i.data(),i.data()+i.size(),num);
                if(err.ec == std::errc::invalid_argument){
                    f.close();
                    throw std::runtime_error("data parse error");
                }
                res.emplace_back(num);
            }
            return res;
        };

        vec_t<int> userInfo;
        userInfo.resize(gItemNum,0);

        while (getline(f,line)){
            auto parsedInfo = parseLine(line);
            curUserId = parsedInfo[0];
            if(curUserId != lastUserId){
                lastUserId = curUserId;
                mData.emplace_back(userInfo);
                userInfo.clear();
                userInfo.resize(gItemNum,0);
            }
            userInfo.at(parsedInfo[1]-1) = parsedInfo[2];
        }
        mData.emplace_back(userInfo);
        userInfo.clear();
        if(mData.size() != gUserNum){
            cout << mData.size() << " vs " << gUserNum;
            f.close();
            throw std::runtime_error("broken dat");
        }
        f.close();
        std::tie(mTrainSet, mTestSet) = divideDataset(mData, 90);

    }
    static std::tuple<mat_t<int>, mat_t<int>> divideDataset[[nodiscard]](const mat_t<int>& data, int pivot){
        if(pivot < 0 || pivot > 100){
            throw std::runtime_error("pivot should in [0, 100]");
        }
        const size_t nItems = data.front().size();
        const size_t nUsers = data.size();
        std::default_random_engine randomEngine;
        std::uniform_int_distribution<int> distribution(0, 99);
        mat_t<int> trainSet, testSet;
        trainSet.reserve(nUsers);
        testSet.reserve(nUsers);

        for(const auto& d: data){
            vec_t<int> trainLine, testLine;
            trainLine.reserve(nItems);
            testSet.reserve(nItems);
            for(auto score: d){
                bool belongTrain = distribution(randomEngine) < pivot;
                if(belongTrain){
                    trainLine.emplace_back(score);
                    testLine.emplace_back(0);
                } else {
                    trainLine.emplace_back(0);
                    testLine.emplace_back(score);
                }
            }
            trainSet.emplace_back(std::move(trainLine));
            testSet.emplace_back(std::move(testLine));
        }
        return {trainSet, testSet};
    }

//    使用Pearson相似度，参考链接：https://blog.csdn.net/duyibo123/article/details/110915485
    static mat_t<double> calculateSimilarity(const mat_t<int>& data){
        if(data.empty()){
            throw std::runtime_error("empty data");
        }
        size_t nUsers = data.size();
        size_t nItems = data.front().size();
        vec_t<double> averageScore;
        averageScore.reserve(nUsers);
        mat_t<double> similarityMatrix(nUsers,vec_t<double>(nUsers,0));
        mat_t<int> scoredFilms;
        scoredFilms.reserve(nUsers);
        for(auto &user : data){
            if(user.size() != nItems){
                throw std::runtime_error("invalid data");
            }
            double sumScore = 0;
            vec_t<int> filmItem;
            for(size_t film=0; film < nItems; film++){
                if(user[film] != 0){
                    sumScore += user[film];
                    filmItem.emplace_back(film);
                }
            }
            averageScore.emplace_back(sumScore/double (filmItem.size()));
            scoredFilms.emplace_back(filmItem);
        }
        for(int user1 = 0 ; user1 < nUsers ; user1++)
            for(int user2 = 0 ; user2 < nUsers ; user2++){
                if(user1 == user2) continue;
                vec_t<int> sameFilm;
                auto &film1 = scoredFilms[user1],&film2 = scoredFilms[user2];
                int i = 0 , j = 0;
                while (i < film1.size() && j < film2.size()){
                    if (film1[i] < film2[j])
                        i++;
                    else if(film1[i] > film2[j])
                        j++;
                    else{
                        sameFilm.emplace_back(film1[i]);
                        i++,j++;
                    }
                }
                double denominator = 0 , numerator1 = 0 , numerator2 = 0;
                for(auto film : sameFilm){
                    auto diff1 = data[user1][film] - averageScore[user1];
                    auto diff2 = data[user2][film] - averageScore[user2];
                    denominator += diff1 * diff2;
                    numerator1 += diff1 * diff1;
                    numerator2 += diff2 * diff2;
                }
                if (numerator1 == 0 || numerator2 == 0)
                    similarityMatrix[user1][user2] = 0;
                else
                    similarityMatrix[user1][user2] = denominator / sqrt(numerator1 * numerator2);
            }
        return similarityMatrix;
    }

    mat_t<std::pair<double, size_t>> getKSimilarMatrix [[nodiscard]](const mat_t<double>& similarMatrix) const{
        const size_t nUsers = similarMatrix.size();
        mat_t<std::pair<double, size_t>> result;
        result.reserve(nUsers);
        for(auto& simUsers: similarMatrix){
            std::vector<std::pair<double, size_t>> kSimUsers;
            kSimUsers.reserve(nUsers);
            for(size_t user=0; user<nUsers; user++){
                kSimUsers.emplace_back(simUsers[user], user);
            }
            std::sort(kSimUsers.begin(), kSimUsers.end(), std::greater<>());
            kSimUsers.resize(mNSimilar);
            result.emplace_back(std::move(kSimUsers));
        }
        return result;
    }

    std::vector<std::pair<double, size_t>> recommend [[nodiscard]](
            const mat_t<std::pair<double, size_t>>& kSimilarMatrix,const int &user) const {
        auto &simUsers = kSimilarMatrix[user];
        auto &userScore = mTrainSet[user];
        const size_t nFilms = userScore.size();
        std::unordered_set<int> watchedFilm;
        for(int film = 0 ; film < nFilms ; film++)
            if(userScore[film]) watchedFilm.insert(film);
        std::vector<std::pair<double, size_t>> relatedFilms;
        std::vector<double> filmScore(nFilms);
        for(auto &[weight,other] : simUsers){
            auto &otherScores = mTrainSet[other];
            for(int film=0; film<nFilms; film++){
                if(watchedFilm.count(film))
                    continue;
                auto score = otherScores[film];
                if(score != 0){
                    filmScore[film] += weight*score;
                }
            }
        }
        for(int i = 0 ; i < nFilms ; i++)
            relatedFilms.emplace_back(filmScore[i],i);
        std::sort(relatedFilms.begin(), relatedFilms.end(), std::greater<>());
        relatedFilms.resize(mNRecommend);
        return relatedFilms;
    }

    double test[[nodiscard]]() const {
        const size_t nUsers = mTrainSet.size();
        size_t hit = 0;
        const auto similarityMatrix = calculateSimilarity(mTrainSet);
        const auto kSimilarMatrix = getKSimilarMatrix(similarityMatrix);
        for(size_t user=0; user<nUsers; user++){
            auto recommendFilms = recommend(kSimilarMatrix, int(user));
            for(auto [w, film]: recommendFilms){
                if(mTestSet[user][film] != 0){
                    hit += 1;
                }
            }
        }
        size_t recommendCount = mNRecommend * nUsers;
        double hitRate = double(hit) / double(recommendCount);
        return hitRate;
    }

};


int main(){
    const char *outputFile = "./output/userCF_result.txt";
    std::cout << "output to '" << outputFile << "'\n";
    freopen(outputFile,"w",stdout);

    std::cout <<"when testing userCF\n";
    auto t0 = std::chrono::system_clock::now();
    UserCF userCf("./data/ratings.dat", 20, 10);
    auto t1 = std::chrono::system_clock::now();
    std::cout << "hit-rate = " << userCf.test() << '\n';
    auto t2 = std::chrono::system_clock::now();

    auto dt1 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()) / 1000;
    auto dt2 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()) / 1000;

    std::cout << dt1 << "s for loading data\n" <<dt2 << "s for testing hit rate\n";
    return 0;
}