- [ ] crop label을 바탕으로 한 StratifiedKFold -> crop StratifiedKFold
    - 각 fold에 crop은 골고루 분포되었지만 risk와 disease가 골고루 분포되지 않아 train 결과는 Good, dacon submission 결과는 BAD가 나온것으로 사료됨
- [x] custom beam search decodern 
    - 0.5 -> 0.8 의 성능 향상
- [ ] crop,disease,risk 따로 분류하지 않고 함께 crop_disease_risk로 분류하기
