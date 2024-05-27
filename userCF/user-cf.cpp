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
#include <numeric>

using namespace std;

vector<string_view> split(const string_view &str, const char s) {
    vector<string_view> result;
    size_t i = 0, j = 0;
    for (; j < str.size(); j++) {
        if (str[j] == s) {
            result.emplace_back(str.substr(i, j - i));
            i = j + 1;
        }
    }
    result.emplace_back(str.substr(i, j - i));
    return result;
}

template<typename T> using vec_t = std::vector<T>;
template<typename T> using mat_t = std::vector<std::vector<T>>;

class UserCF {
private:
    const int mNSimilar, mNRecommend;
    mat_t<int> mData;
    mat_t<int> mTrainSet, mTestSet;
    mat_t<double> mTrainSimilarMatrix;
    static const int gUserNum = 6040;
    static const int gItemNum = 3952;

public:
    UserCF(const std::string &dataPath, int nSimilar, int nRecommend) :
            mNSimilar(nSimilar), mNRecommend(nRecommend) {
        std::ifstream f(dataPath, std::ios::in);
        if (!f.good()) {
            throw std::runtime_error("open file '" + dataPath + "' error");
        }
        std::string line;
        int lastUserId = 1;
        int curUserId;
        auto parseLine = [&](const std::string &cur) -> vec_t<int> {
            auto datItem = split(cur, ':');
            vec_t<int> res;
            for (auto i: datItem) {
                int num;
                if (i.empty())
                    continue;
                auto err = std::from_chars(i.data(), i.data() + i.size(), num);
                if (err.ec == std::errc::invalid_argument) {
                    f.close();
                    throw std::runtime_error("data parse error");
                }
                res.emplace_back(num);
            }
            return res;
        };

        vec_t<int> userInfo(gItemNum, 0);

        while (getline(f, line)) {
            auto parsedInfo = parseLine(line);
            curUserId = parsedInfo[0];
            if (curUserId != lastUserId) {
                lastUserId = curUserId;
                mData.emplace_back(userInfo);
                fill(userInfo.begin(), userInfo.end(), 0);
            }
            userInfo.at(parsedInfo[1] - 1) = parsedInfo[2];
        }
        mData.emplace_back(userInfo);
        if (mData.size() != gUserNum) {
            cout << mData.size() << " vs " << gUserNum;
            f.close();
            throw std::runtime_error("broken dat");
        }
        f.close();
        std::tie(mTrainSet, mTestSet) = divideDataset(mData, 90);
    }

    static std::tuple<mat_t<int>, mat_t<int>> divideDataset(const mat_t<int> &data, int pivot) {
        if (pivot < 0 || pivot > 100) {
            throw std::runtime_error("pivot should in [0, 100]");
        }
        const size_t nItems = data.front().size();
        const size_t nUsers = data.size();
        std::default_random_engine randomEngine;
        std::uniform_int_distribution<int> distribution(0, 99);
        mat_t<int> trainSet, testSet;
        trainSet.reserve(nUsers);
        testSet.reserve(nUsers);

        for (const auto &d: data) {
            vec_t<int> trainLine(nItems, 0), testLine(nItems, 0);
            for (size_t i = 0; i < nItems; i++) {
                if (distribution(randomEngine) < pivot) {
                    trainLine[i] = d[i];
                } else {
                    testLine[i] = d[i];
                }
            }
            trainSet.emplace_back(std::move(trainLine));
            testSet.emplace_back(std::move(testLine));
        }
        return {trainSet, testSet};
    }

    static mat_t<double> calculateSimilarity(const mat_t<int> &data) {
        if (data.empty()) {
            throw std::runtime_error("empty data");
        }
        size_t nUsers = data.size();
        size_t nItems = data.front().size();
        mat_t<double> similarityMatrix(nUsers, vec_t<double>(nUsers, 0));

        // 计算用户间的皮尔逊相似度
        [[maybe_unused]] auto computeUserPearsonSimilarity = [&](size_t user1, size_t user2) {
            // 计算每个用户的平均分和评分过的电影列表
            vec_t<double> averageScore(nUsers, 0);
            mat_t<int> scoredFilms(nUsers);
            for (size_t user = 0; user < nUsers; user++) {
                double sumScore = 0;
                int count = 0;
                for (size_t film = 0; film < nItems; film++) {
                    if (data[user][film] != 0) {
                        sumScore += data[user][film];
                        scoredFilms[user].emplace_back(film);
                        count++;
                    }
                }
                averageScore[user] = count > 0 ? sumScore / count : 0;
            }
            const auto &films1 = scoredFilms[user1];
            const auto &films2 = scoredFilms[user2];
            vec_t<int> commonFilms;
            set_intersection(films1.begin(), films1.end(), films2.begin(), films2.end(), back_inserter(commonFilms));

            double numerator = 0, denominator1 = 0, denominator2 = 0;
            for (const auto &film: commonFilms) {
                double diff1 = data[user1][film] - averageScore[user1];
                double diff2 = data[user2][film] - averageScore[user2];
                numerator += diff1 * diff2;
                denominator1 += diff1 * diff1;
                denominator2 += diff2 * diff2;
            }

            double similarity = (denominator1 == 0 || denominator2 == 0) ? 0 : numerator / sqrt(denominator1 * denominator2);
            similarityMatrix[user1][user2] = similarity;
            similarityMatrix[user2][user1] = similarity;
        };

        // 计算用户间的余弦相似度
        [[maybe_unused]] auto computeUserCosineSimilarity = [&](size_t user1, size_t user2) {
            double numerator = 0, denominator1 = 0, denominator2 = 0;
            for (size_t film = 0; film < nItems; film++) {
                double score1 = data[user1][film];
                double score2 = data[user2][film];
                numerator += score1 * score2;
                denominator1 += score1 * score1;
                denominator2 += score2 * score2;
            }

            double similarity = (denominator1 == 0 || denominator2 == 0) ? 0 : numerator / (sqrt(denominator1) * sqrt(denominator2));
            similarityMatrix[user1][user2] = similarity;
            similarityMatrix[user2][user1] = similarity;
        };

        for (size_t user1 = 0; user1 < nUsers; user1++) {
            for (size_t user2 = user1 + 1; user2 < nUsers; user2++) {
                computeUserCosineSimilarity(user1,user2);
            }
        }

        return similarityMatrix;
    }


    [[nodiscard]] mat_t<std::pair<double, size_t>> getKSimilarMatrix(const mat_t<double> &similarMatrix) const {
        const size_t nUsers = similarMatrix.size();
        mat_t<std::pair<double, size_t>> result(nUsers);

        for (size_t user = 0; user < nUsers; user++) {
            std::vector<std::pair<double, size_t>> kSimUsers;
            for (size_t other = 0; other < nUsers; other++) {
                if (user != other) {
                    kSimUsers.emplace_back(similarMatrix[user][other], other);
                }
            }
            std::partial_sort(kSimUsers.begin(), kSimUsers.begin() + mNSimilar, kSimUsers.end(), std::greater<>());
            result[user] = vec_t<std::pair<double, size_t>>(kSimUsers.begin(), kSimUsers.begin() + mNSimilar);
        }
        return result;
    }

    [[nodiscard]] std::vector<std::pair<double, size_t>> recommend(const mat_t<std::pair<double, size_t>> &kSimilarMatrix, const int &user) const {
        auto &simUsers = kSimilarMatrix[user];
        auto &userScore = mTrainSet[user];
        const size_t nFilms = userScore.size();
        std::unordered_set<int> watchedFilm;
        for (int film = 0; film < nFilms; film++)
            if (userScore[film]) watchedFilm.insert(film);
        std::vector<double> filmScore(nFilms);
        for (auto &[weight, other]: simUsers) {
            auto &otherScores = mTrainSet[other];
            for (int film = 0; film < nFilms; film++) {
                if (watchedFilm.count(film))
                    continue;
                auto score = otherScores[film];
                if (score != 0) {
                    filmScore[film] += weight * score;
                }
            }
        }
        std::vector<std::pair<double, size_t>> relatedFilms;
        for (int i = 0; i < nFilms; i++)
            relatedFilms.emplace_back(filmScore[i], i);
        std::partial_sort(relatedFilms.begin(), relatedFilms.begin() + mNRecommend, relatedFilms.end(), std::greater<>());
        relatedFilms.resize(mNRecommend);
        return relatedFilms;
    }

    [[nodiscard]] double test() const {
        const size_t nUsers = mTrainSet.size();
        size_t hit = 0;
        const auto similarityMatrix = calculateSimilarity(mTrainSet);
        const auto kSimilarMatrix = getKSimilarMatrix(similarityMatrix);
        for (size_t user = 0; user < nUsers; user++) {
            auto recommendFilms = recommend(kSimilarMatrix, int(user));
            for (auto [w, film]: recommendFilms) {
                if (mTestSet[user][film] != 0) {
                    hit += 1;
                }
            }
        }
        size_t recommendCount = mNRecommend * nUsers;
        double hitRate = double(hit) / double(recommendCount);
        return hitRate;
    }
};

int main() {
    const char *outputFile = "./output/userCF_result.txt";
    std::cout << "output to '" << outputFile << "'\n";
    freopen(outputFile, "w", stdout);

    std::cout << "when testing userCF\n";
    auto t0 = std::chrono::system_clock::now();
    std::cout <<"nSimilar : 50 , nRecommend : 5\n";
    UserCF userCf("./data/ratings.dat", 50, 5);

    auto t1 = std::chrono::system_clock::now();
    std::cout << "hit-rate = " << userCf.test() << '\n';
    auto t2 = std::chrono::system_clock::now();

    auto dt1 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) / 1000;
    auto dt2 = double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000;

    std::cout << dt1 << "s for loading data\n" << dt2 << "s for testing hit rate\n";
    return 0;
}
