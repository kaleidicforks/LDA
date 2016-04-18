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

#ifndef _OPENSOURCE_GLDA_SAMPLER_H__
#define _OPENSOURCE_GLDA_SAMPLER_H__

#include "common.hh"
#include "document.hh"
#include "model.hh"
#include "accumulative_model.hh"

namespace learning_lda {

// LDASampler trains LDA models and computes statistics about documents in
// LDA models.
class LDASampler {
 public:
  // alpha and beta are the Gibbs sampling symmetric hyperparameters.
  // model is the model to use.
  LDASampler(double alpha, double beta,
             LDAModel* model,
             LDAAccumulativeModel* accum_model);

  ~LDASampler() {}

  // Given a corpus, whose every word has been initialized (i.e.,
  // assigned a random topic), this function initializes model_ to
  // count the word-topic co-occurrences.
  void InitializeModel(const LDACorpus& corpus);

  void UpdateModel(const LDACorpus& corpus);

  // Performs one round of Gibbs sampling on documents in the corpus
  // by invoking DoGibbsSampling(...).  If we are to train
  // a model given training data, we should set update_model to true,
  // and the algorithm updates model_ during Gibbs sampling.
  // Otherwise, if we are to sample the latent topics of a query
  // document, we should set update_model to false.  If update_model is
  // true, burn_in indicates should we accumulate the current estimate
  // to accum_model_.  For the first certain number of iterations,
  // where the algorithm has not converged yet, you should set burn_in
  // to false.  After that, we should set burn_in to true.
  void DoGibbsSampling(LDACorpus* corpus, bool update_model, bool burn_in);

  // Performs one round of Gibbs sampling on a document.  Updates
  // document's topic assignments.  For learning, update_model_=true,
  // for sampling topics of a query, update_model_==false.
  void DoGibbsSampling(LDADocument* document, bool update_model);


  // Computes the log likelihood of a document.
  double ComputeLogLikelihood(const LDACorpus& corpus) const;
  double ComputeLogLikelihood(LDADocument* document) const;

 private:
  const double alpha_;
  const double beta_;
  LDAModel* model_;
  LDAAccumulativeModel* accum_model_;

  // The core of the Gibbs sampling process.  Compute the full conditional
  // posterior distribution of topic assignments to the indicated word.
  //
  // That is, holding all word-topic assignments constant, except for the
  // indicated one, compute a non-normalized probability distribution over
  // topics for the indicated word occurrence.
  void GenerateTopicDistributionForWord(const LDADocument& document,
      int word, int current_word_topic, bool update_model,
      vector<double>* distribution) const;
};


}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_SAMPLER_H__
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

#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

#include <math.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>

#include "sampler.hh"
#include "document.hh"
#include "model.hh"

namespace learning_lda {

  using namespace std;

  LDASampler::LDASampler(double alpha,
                         double beta,
                         LDAModel* model,
                         LDAAccumulativeModel* accum_model)
    : alpha_(alpha), beta_(beta), model_(model), accum_model_(accum_model) {
    CHECK_LT(0.0, alpha);
    CHECK_LT(0.0, beta);
    CHECK(model != NULL);
  }

  void LDASampler::InitializeModel(const LDACorpus& corpus) {
#pragma omp parallel for
    for (int i = 0; i < corpus.size(); ++i) {
      LDADocument* document = corpus[i];
      for (LDADocument::WordOccurrenceIterator iter2(document);
           !iter2.Done();
           iter2.Next()) {
        model_->IncrementTopic(iter2.Word(), iter2.Topic(), 1);
      }
    }
  }

  void LDASampler::UpdateModel(const LDACorpus& corpus) {
    // Clear all counts, but do not change the structure of the model.
    model_->Unset();
    // and re-fill the model.
    InitializeModel(corpus);
  }

  void LDASampler::DoGibbsSampling(LDACorpus* corpus,
                                   bool update_model,
                                   bool burn_in) {
#pragma omp parallel for
    for (int i = 0; i < corpus->size(); ++i) {
      DoGibbsSampling((*corpus)[i], update_model);
    }

    if (accum_model_ != NULL && update_model && !burn_in) {
      accum_model_->AccumulateModel(*model_);
    }
  }

  void LDASampler::DoGibbsSampling(LDADocument* document,
                                   bool update_model) {
    for (LDADocument::WordOccurrenceIterator iterator(document);
         !iterator.Done();
         iterator.Next()) {
      // This is a (non-normalized) probability distribution from which we will
      // select the new topic for the current word occurrence.
      vector<double> new_topic_distribution;
      GenerateTopicDistributionForWord(*document,
                                       iterator.Word(),
                                       iterator.Topic(),
                                       update_model,
                                       &new_topic_distribution);
      int new_topic = GetAccumulativeSample(new_topic_distribution);
      if (new_topic != -1) {
        // If new_topic != -1 (i.e. GetAccumulativeSample) runs OK, we
        // update document and model parameters with the new topic.
        if (update_model) {
          model_->ReassignTopic(iterator.Word(), iterator.Topic(), new_topic);
        }
        iterator.SetTopic(new_topic);
      }
    }
  }

  void
  LDASampler::GenerateTopicDistributionForWord(const LDADocument& document,
                                               int word,
                                               int current_word_topic,
                                               bool update_model,
                                               vector<double>* distribution)
    const {
    int num_topics = model_->num_topics();
    int num_words = model_->num_words();
    distribution->clear();
    distribution->reserve(num_topics);

    const TopicCountDistribution& word_distribution =
      model_->GetWordTopicDistribution(word);
    for (int k = 0; k < num_topics; ++k) {
      // We will need to temporarily unassign the word from its old
      // topic, which we accomplish by decrementing the appropriate
      // counts by 1.
      int current_topic_adjustment =
        (update_model && k == current_word_topic) ? -1 : 0;
      double topic_word_factor =
        word_distribution[k] + current_topic_adjustment;
      double global_topic_factor =
        model_->GetGlobalTopicDistribution()[k] + current_topic_adjustment;
      double document_topic_factor =
        document.topic_distribution()[k] + current_topic_adjustment;
      distribution->push_back((topic_word_factor + beta_) *
                              (document_topic_factor + alpha_) /
                              (global_topic_factor + num_words * beta_));
    }
  }

  double LDASampler::ComputeLogLikelihood(const LDACorpus& corpus) const {
    vector<double> local_loglikelihood(corpus.size(), 0.0);

#pragma omp parallel for
    for (int i = 0; i < corpus.size(); ++i) {
      local_loglikelihood[i] = ComputeLogLikelihood(corpus[i]);
    }

    return accumulate(local_loglikelihood.begin(), local_loglikelihood.end(),
                      0.0, plus<double>());
  }

  // Compute log P(d) = sum_w log P(w), where P(w) = sum_z P(w|z)P(z|d).
  double LDASampler::ComputeLogLikelihood(LDADocument* document) const {
    const int num_topics(model_->num_topics());

    // Compute P(z|d) for the given document and all topics.
    const vector<int64>& document_topic_cooccurrences(document->
                                                      topic_distribution());
    CHECK_EQ(num_topics, document_topic_cooccurrences.size());
    int64 document_length = 0;
    for (int t = 0; t < num_topics; ++t) {
      document_length += document_topic_cooccurrences[t];
    }
    vector<double> prob_topic_given_document(num_topics);
    for (int t = 0; t < num_topics; ++t) {
      prob_topic_given_document[t] =
        (document_topic_cooccurrences[t] + alpha_) /
        (document_length + alpha_ * num_topics);
    }

    // Get global topic occurrences, which will be used compute P(w|z).
    TopicCountDistribution
      global_topic_occurrences(model_->GetGlobalTopicDistribution());

    double log_likelihood = 0.0;
    // A document's log-likelihood is the sum of log-likelihoods of
    // its words.  Compute the likelihood for every word and sum the
    // logs.
    for (LDADocument::WordOccurrenceIterator iterator(document);
         !iterator.Done();
         iterator.Next()) {
      // Get topic_count_distribution of the current word, which will be
      // used to Compute P(w|z).
      TopicCountDistribution
        word_topic_cooccurrences(model_->
                                 GetWordTopicDistribution(iterator.Word()));

      // Comput P(w|z).
      vector<double> prob_word_given_topic(num_topics);
      for (int t = 0; t < num_topics; ++t) {
        prob_word_given_topic[t] =
          (word_topic_cooccurrences[t] + beta_) /
          (global_topic_occurrences[t] + model_->num_words() * beta_);
      }

      // Compute P(w) = sum_z P(w|z)P(z|d)
      double prob_word = 0.0;
      for (int t = 0; t < num_topics; ++t) {
        prob_word += prob_word_given_topic[t] * prob_topic_given_document[t];
      }

      log_likelihood += log(prob_word);
    }
    return log_likelihood;
  }

}  // namespace learning_lda
