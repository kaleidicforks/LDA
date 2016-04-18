// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _OPENSOURCE_GLDA_DOCUMENT_H__
#define _OPENSOURCE_GLDA_DOCUMENT_H__

#include <string>
#include <utility>
#include <vector>

#include "common.hh"

namespace learning_lda {

class DocumentWordTopicsPB;

// Stores a document as a bag of words and provides methods for interacting
// with Gibbs LDA models.
class LDADocument {
 public:
  // An iterator over all of the word occurrences in a document.
  class WordOccurrenceIterator {
   public:
    // Intialize the WordOccurrenceIterator for a document.
    explicit WordOccurrenceIterator(LDADocument* parent);
    ~WordOccurrenceIterator();

    // Returns true if we are done iterating.
    bool Done();

    // Advances to the next word occurrence.
    void Next();

    // Returns the topic of the current occurrence.
    int Topic();

    // Changes the topic of the current occurrence.
    void SetTopic(int new_topic);

    // Returns the word of the current occurrence.
    int Word();

   private:
    // If the current word has no occurrences, advance until reaching a word
    // that does have occurrences or the end of the document.
    void SkipWordsWithoutOccurrences();

    LDADocument* parent_;
    int word_index_;
    int word_topic_index_;
  };
  friend class WordOccurrenceIterator;

  // Initializes a document from a DocumentWordTopicsPB. Usually, this
  // constructor is used in training an LDA model, because the
  // initialization phase creates the model whose vocabulary covers
  // all words appear in the training data.
  LDADocument(const DocumentWordTopicsPB& topics, int num_topics);

  virtual ~LDADocument();

  // Returns the document's topic associations.
  const DocumentWordTopicsPB& topics() const {
    return *topic_assignments_;
  }

  // Returns the document's topic occurrence counts.
  const vector<int64>& topic_distribution() const {
    return topic_distribution_;
  }

  void ResetWordIndex(const map<string, int>& word_index_map);

  string DebugString();
 protected:
  DocumentWordTopicsPB*  topic_assignments_;
  vector<int64> topic_distribution_;

  // Count topic occurrences in topic_assignments_ and stores the
  // result in topic_distribution_.
  void CountTopicDistribution();
};

typedef vector<LDADocument*> LDACorpus;

}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_DOCUMENT_H__
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "document.hh"
#include <cstdio>

namespace learning_lda {

LDADocument::WordOccurrenceIterator::WordOccurrenceIterator(
    LDADocument* parent) {
  parent_ = parent;
  word_index_ = 0;
  word_topic_index_ = 0;

  SkipWordsWithoutOccurrences();
}

LDADocument::WordOccurrenceIterator::~WordOccurrenceIterator() { }

// Have we advanced beyond the last word?
bool LDADocument::WordOccurrenceIterator::Done() {
  CHECK_GE(parent_->topic_assignments_->words_size(), word_index_);
  return word_index_ == parent_->topic_assignments_->words_size();
}

// We iterate over all the occurrences of each word.  If we have finished with
// the current word, we advance to the 0th ocurrence of the next word that has
// occurrences.
void LDADocument::WordOccurrenceIterator::Next() {
  CHECK(!Done());
  ++word_topic_index_;
  if (word_topic_index_ >
      parent_->topic_assignments_->word_last_topic_index(word_index_)) {
    ++word_index_;
    SkipWordsWithoutOccurrences();
  }
}

int LDADocument::WordOccurrenceIterator::Topic() {
  CHECK(!Done());
  return parent_->topic_assignments_->wordtopics(word_topic_index_);
}

// Exchange the topic.  Be sure to keep the topic count distribution up to
// date.
void LDADocument::WordOccurrenceIterator::SetTopic(int new_topic) {
  CHECK(!Done());
  CHECK_LE(0, new_topic);
  CHECK_GT(parent_->topic_distribution_.size(), new_topic);
  // Adjust the topic counts before we set the new topic and forget the old
  // one.
  parent_->topic_distribution_[Topic()] -= 1;
  parent_->topic_distribution_[new_topic] += 1;
  *(parent_->topic_assignments_->mutable_wordtopics(word_topic_index_)) =
    new_topic;
}

int LDADocument::WordOccurrenceIterator::Word() {
  CHECK(!Done());
  return parent_->topic_assignments_->word(word_index_);
}

void LDADocument::WordOccurrenceIterator::SkipWordsWithoutOccurrences() {
  // The second part of the condition means "while the current word has no
  // occurrences" (and thus no topic assignments).
  while (
      !Done() &&
      parent_->topic_assignments_->wordtopics_count(word_index_) == 0) {
    ++word_index_;
  }
}

void LDADocument::CountTopicDistribution() {
  for (int i = 0; i < topic_distribution_.size(); ++i) {
    topic_distribution_[i] = 0;
  }
  for (WordOccurrenceIterator iter(this); !iter.Done(); iter.Next()) {
    topic_distribution_[iter.Topic()] += 1;
  }
}

string LDADocument::DebugString() {
  string s;
  for (int i = 0; i < topic_assignments_->wordtopics_.size(); ++i) {
    char buf[100];
    snprintf(buf, sizeof(buf), "%d", topic_assignments_->wordtopics_[i]);
    s.append(buf);
    s.append(" ");
  }
  s.append("#");
  for (int i = 0; i < topic_distribution_.size(); ++i) {
    char buf[100];
    snprintf(buf, sizeof(buf), "%lld", topic_distribution_[i]);
    s.append(buf);
    s.append(" ");
  }
  return s;
}

LDADocument::LDADocument(const DocumentWordTopicsPB& topics,
                         int num_topics) {
  topic_assignments_ = new DocumentWordTopicsPB;
  topic_assignments_->CopyFrom(topics);

  topic_distribution_.resize(num_topics);
  CountTopicDistribution();
}

LDADocument::~LDADocument() {
  delete topic_assignments_;
  topic_assignments_ = NULL;
}

void LDADocument::ResetWordIndex(const map<string, int>& word_index_map) {
  for (int i = 0; i < topic_assignments_->words_.size(); ++i) {
    (*topic_assignments_).words_[i] =
      word_index_map.find((*topic_assignments_).words_s_[i])->second;
  }
}
}  // namespace learning_lda
