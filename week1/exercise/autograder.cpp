#include <bits/stdc++.h>
using namespace std;

/*
    Compile: g++ -O2 autograder_pick12.cpp -o autograder
    Run: ./autograder

    Students should fill only the solve function below.
*/

string solve(int n, vector<long long> a) {
    // TODO: Fill this function.
    // Return one of: "Player 1" or "Player 2" or "Draw"
    int differences[n];
    for(int i=0;i<n;i++){
        differences[i]=0;
    }
    differences[n-1]=a[n-1];
    differences[n-2]=max(a[n-2]-a[n-1],a[n-2]+a[n-1]);
    for(int i=n-3;i>=0;i--){
        differences[i]=max(a[i]-differences[i+1],a[i]+a[i+1]-differences[i+2]);
    }
    if (differences[0]>0){return "Player 1";}
    else if (differences[0]<0){return "Player 2";}
    return "Draw";
}

static string trim(const string &s) {
    int l = 0, r = (int)s.size() - 1;
    while (l <= r && isspace((unsigned char)s[l])) l++;
    while (r >= l && isspace((unsigned char)s[r])) r--;
    if (l > r) return "";
    return s.substr(l, r - l + 1);
}

static bool file_exists(const string &path) {
    ifstream f(path);
    return f.good();
}

int main() {
    const string folder = "testcases";

    int passed = 0;
    int total = 0;

    for (int tc = 0; ; tc++) {
        string in_path = folder + "/input" + to_string(tc) + ".txt";
        string out_path = folder + "/output" + to_string(tc) + ".txt";

        if (!file_exists(in_path) && !file_exists(out_path)) {
            break;
        }

        total++;

        if (!file_exists(in_path)) {
            cout << "Failed on testcase " << tc << ": missing " << in_path << '\n';
            continue;
        }
        if (!file_exists(out_path)) {
            cout << "Failed on testcase " << tc << ": missing " << out_path << '\n';
            continue;
        }

        ifstream fin(in_path);
        int n;
        fin >> n;
        vector<long long> a(n);
        for (int i = 0; i < n; i++) fin >> a[i];

        if (!fin) {
            cout << "Failed on testcase " << tc << ": invalid input format\n";
            continue;
        }

        string got = trim(solve(n, a));

        ifstream fout(out_path);
        string expected, line;
        while (getline(fout, line)) {
            if (!expected.empty()) expected += "\n";
            expected += line;
        }
        expected = trim(expected);

        if (got == expected) {
            cout << "OK testcase " << tc << '\n';
            passed++;
        } else {
            cout << "Failed on testcase " << tc << '\n';
            cout << "Expected: " << expected << '\n';
            cout << "Got     : " << got << '\n';
        }
    }

    if (total == 0) {
        cout << "No testcases found. Make sure there is a testcases folder with input0.txt and output0.txt.\n";
        return 1;
    }

    cout << "\nPassed " << passed << " / " << total << " testcases.\n";

    if (passed == total) {
        cout << "OK\n";
        return 0;
    } else {
        cout << "Failed\n";
        return 1;
    }
}
