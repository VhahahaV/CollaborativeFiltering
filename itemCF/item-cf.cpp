#include <iostream>
#include <fstream>
#include <vector>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <chrono>
#include <stdio.h>
///refer:git@github.com:fangtiancheng/LargeScaleDataProcessSJTU2024.git
using namespace std;
std::vector<std::string_view> split(const std::string_view& str, const char s){
    std::vector<std::string_view> result;
    int i=0, j=0;
    for(; j<str.size(); j++){
        if(str[j] == s){
            result.emplace_back(str.substr(i, j-i));
            i = j+1;
        }
    }
    result.emplace_back(str.substr(i, j-i));
    return result;
}

template<typename T> using vec_t = std::vector<T>;
template<typename T> using mat_t = std::vector<std::vector<T>>;
template<typename T>
std::ostream& saveCsv(std::ostream& f, const mat_t<T>& scores){
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

class ItemCF{
private:
    const int mNSimilar, mNRecommend;
    mat_t<int> mData;
    mat_t<int> mTrainSet, mTestSet;
    mat_t<double> mTrainSimilarMatrix;
    static const int gUserNum = 6040;
    static const int gItemNum =3952;
public:
    ItemCF(const std::string& dataPath, int nSimilar, int nRecommend):
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

    static mat_t<double> calculateSimilarity(const mat_t<int>& data){
        if(data.empty()){
            throw std::runtime_error("empty data");
        }
        size_t nItems = data.front().size();
        vec_t<int> popularItems(nItems, 0);
        mat_t<double> similarityMatrix(nItems);
        for(auto& m: similarityMatrix) m.resize(nItems, 0);
        for(auto& user: data){
            if(user.size() != nItems){
                throw std::runtime_error("invalid data");
            }
            std::vector<size_t> scoredFilms;
            for(size_t film=0; film < nItems; film++){
                if(user[film] != 0){
                    popularItems[film] += 1;
                    scoredFilms.emplace_back(film);
                }
            }
            for(auto film1: scoredFilms){
                for(auto film2: scoredFilms){
                    if(film1 != film2){
                        similarityMatrix[film1][film2] += 1;
                    }
                }
            }
        }
        for(int i=0; i<nItems; i++){
            for(int j=0; j<nItems; j++){
                if(similarityMatrix[i][j] != 0){
                    similarityMatrix[i][j] /= std::sqrt(double(popularItems[i]*popularItems[j]));
                }
            }
        }
        return similarityMatrix;
    }

    mat_t<std::pair<double, size_t>> getKSimilarMatrix [[nodiscard]](const mat_t<double>& similarMatrix) const{
        const size_t nFilms = similarMatrix.size();
        mat_t<std::pair<double, size_t>> result;
        result.reserve(nFilms);
        for(auto& simFilms: similarMatrix){
            std::vector<std::pair<double, size_t>> kSimFilms;
            kSimFilms.reserve(nFilms);
            for(size_t sfilm=0; sfilm<nFilms; sfilm++){
                kSimFilms.emplace_back(simFilms[sfilm], sfilm);
            }
            std::sort(kSimFilms.begin(), kSimFilms.end(), std::greater<>());
            kSimFilms.resize(mNSimilar);
            result.emplace_back(std::move(kSimFilms));
        }
        return result;
    }

    std::vector<std::pair<double, size_t>> recommend [[nodiscard]](
            const mat_t<std::pair<double, size_t>>& kSimilarMatrix, const vec_t<int>& userScore) const {
        const size_t nFilms = userScore.size();
        std::vector<size_t> watchedFilms;
        for(size_t film=0; film<nFilms; film++){
            if(userScore[film] != 0){
                watchedFilms.emplace_back(film);
            }
        }
        std::vector<std::pair<double, size_t>> relatedFilms;
        relatedFilms.reserve(nFilms);
        for(size_t film=0; film<nFilms; film++){
            relatedFilms.emplace_back(0, film);
        }
        for(auto film: watchedFilms){
            auto rating = userScore[film];
            const auto& kSimFilms = kSimilarMatrix[film];
            for(auto [w, sfilm]: kSimFilms){
                if(std::binary_search(watchedFilms.cbegin(), watchedFilms.cend(), sfilm))
                    continue;
                relatedFilms[sfilm].first += (w*rating);
            }
        }
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
            auto recommendFilms = recommend(kSimilarMatrix, mTrainSet[user]);
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

    const char *outputFile = "./output/itemCF_result.txt";
    std::cout << "output to '" << outputFile << "'\n";
    freopen(outputFile,"w",stdout);
    std::cout <<"when testing itemCF\n";
    auto t0 = std::chrono::system_clock::now();
    ItemCF itemCf("./data/ratings.dat", 20, 10);
    auto t1 = std::chrono::system_clock::now();
    std::cout << "hit-rate = " << itemCf.test() << '\n';
    auto t2 = std::chrono::system_clock::now();

    auto dt1 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()) / 1000;
    auto dt2 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()) / 1000;

    std::cout << dt1 << "s for loading data\n" <<dt2 << "s for testing hit rate\n";
    return 0;
}