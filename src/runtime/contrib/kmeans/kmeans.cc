#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <vector>
#include <atomic>
#include <cstring>
#include <tvm/support/parallel_for.h>


namespace tvm {
namespace contrib {
  using namespace runtime;
  template<typename T>
  void atomic_add(std::atomic<T>& ptr, T amt) {
    T value = ptr.load();
    T new_value = value + amt;
    while(!ptr.compare_exchange_weak(value, new_value)) {
      value = ptr.load();
      new_value = value + amt;
    }
  }

  template<typename DSDType>
  void kmeans_aggregate(DLTensor* dataset, DLTensor* center_indexes, DLTensor* distances,
			DLTensor* new_centers, DLTensor* centroid_counts, DLTensor* centroid_scores) {
    int n_rows = dataset->shape[0];
    int n_cols = dataset->shape[1];
    int n_centers = new_centers->shape[0];
    int check_cols = new_centers->shape[1];
    CHECK_EQ(n_cols,check_cols) << "Column count mismatch: dataset - "
				<< n_cols << " centroids - " << check_cols;
    DSDType* dataset_data = (DSDType*)dataset->data;
    int* center_index_data = (int*)center_indexes->data;
    DSDType* distance_data = (DSDType*)distances->data;

    std::memset(new_centers->data, 0, n_centers * n_cols * sizeof(double));
    std::atomic<double>* center_data = reinterpret_cast<std::atomic<double>*>(new_centers->data);

    std::memset(centroid_counts->data, 0, n_centers * sizeof(long));
    std::atomic<long>* count_data = reinterpret_cast<std::atomic<long>*>(centroid_counts->data);

    std::memset(centroid_scores->data, 0, n_centers * sizeof(double));
    std::atomic<double>* centroid_scores_data = reinterpret_cast<std::atomic<double>*>(centroid_scores->data);

    support::parallel_for(0, n_rows, [=](int row_idx) {
      auto center_idx = center_index_data[row_idx];
      for(int col_idx = 0; col_idx < n_cols; ++col_idx) {
	atomic_add(center_data[center_idx * n_cols  + col_idx], (double)dataset_data[row_idx*n_cols + col_idx]);
      }
      atomic_add(count_data[center_idx], (long)1);
      atomic_add(centroid_scores_data[center_idx], (double)distance_data[row_idx]);
    });
  }

TVM_REGISTER_GLOBAL("tvm.contrib.kmeans.aggregate").set_body([](TVMArgs args, TVMRetValue* ret) {
  kmeans_aggregate<float>(args[0], args[1], args[2], args[3], args[4], args[5]);
});

}}
