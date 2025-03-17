#include <iostream>
#include <string>
#include <vector>

#include "permutations_cxx.h"

int main() {
  std::vector<std::string> words = {"aaa", "acb", "acd", "ad",  "adc", "bac",
                                    "bc",  "bcc", "bd",  "bda", "bdc", "caa",
                                    "cad", "cb",  "cc",  "ccb", "cd",  "dac",
                                    "db",  "dc",  "dca", "dcb", "dcc", "dd"};

  dictionary_t dictionary;

  for (auto word : words)
    dictionary[word];

  Permutations(dictionary);

  for (auto entry : dictionary) {
    std::cout << entry.first << " : ";
    for (auto word : entry.second)
      std::cout << word << " ";
    std::cout << "\n";
  }

  return 0;
}