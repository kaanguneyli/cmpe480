import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class DecisionTree():
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth if max_depth is not None else float('inf')  # maximum depth of the tree
        self.min_samples_split = min_samples_split  # minimum number of samples required to split an internal node
        
        # a recursive dictionary that represents the decision tree
        # has 4 keys: 'leaf', 'label', 'split_feature', 'children'
        # these 4 keys represent only the first node of the tree
        # other nodes are stored in the 'children' key recursively
        self._tree = None  

    def __str__(self):
        if self._tree is None:
            return "Decision tree has not been fitted yet."

        return self._tree_to_str(self._tree, indent="")


    def _tree_to_str(self, node, indent=""):
        if node['leaf']:
            return f"{indent}Leaf: predicts {node['label']}\n"   # return the label of the leaf node

        # if the node is not a leaf node, print the split feature and recursively print the children
        # also create a tree-like structure with indentations
        s = f"{indent}Split on feature: {node['split_feature']}\n"
        for val, child_node in node['children'].items():
            s += f"{indent}    Value = {val}:\n"
            s += self._tree_to_str(child_node, indent + "        ")
        return s


    def fit(self, X, y):
        self._tree = self._fit(X, y, depth=0)


    def _fit(self, X, y, depth):
        # return a leaf node if the stopping criteria are met (base case)

        # base case 1: if all the labels are the same, return a leaf node
        if len(np.unique(y)) == 1:
            return {
                'leaf': True,
                'label': y.iloc[0],
                'depth': depth
            }
        
        # base case 2: if there are no features left to split on, return a leaf node
        if len(X.columns) == 0:
            return {
                'leaf': True,
                'label': y.value_counts().idxmax(),
                'depth': depth
            }
        
        # base case 3: if the maximum depth is reached, return a leaf node
        if depth >= self.max_depth:
            return {
                'leaf': True,
                'label': y.value_counts().idxmax(),
                'depth': depth
            }

        # base case 4: if the number of samples is less than the minimum samples required to split, return a leaf node
        if len(y) < self.min_samples_split:
            return {
                'leaf': True,
                'label': y.value_counts().idxmax(),
                'depth': depth
            }
        
        current_entropy = self._entropy(y)   # calculate the entropy of the current node
        
        best_info_gain = 0
        best_feature = None
        best_X_list = []
        best_y_list = []

        # iterate over all the features and find the one that gives the best information gain
        for feature in X.columns:
            X_list, y_list = self._split_feature(X, y, feature)
            info_gain = self._information_gain(current_entropy, y, y_list)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_X_list = X_list
                best_y_list = y_list

        # base case 5: if the best information gain is 0 (no improvement on the model), return a leaf node
        if best_info_gain == 0:
            return {
                'leaf': True,
                'label': y.value_counts().idxmax(),
                'depth': depth
            }
        
        depth += 1  # increment the depth of the tree, since we are going to split on a feature

        # create a node with the best feature to split on
        node = {
            'leaf': False,
            'split_feature': best_feature,
            'children': {},
            'label': y.value_counts().idxmax(),  # non-leaf nodes have labels as well (most common label) as a fallout plan
            'depth': depth
        }
        
        # recursively create the children of the node
        unique_values = np.unique(X[best_feature])
        for i, val in enumerate(unique_values):
            child_subtree = self._fit(best_X_list[i], best_y_list[i], depth)
            node['children'][val] = child_subtree

        return node


    def predict(self, X):
        # apply the _predict_one function to each row of the dataframe
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._predict_one(row))
        return pd.Series(predictions, index=X.index)

    
    def _predict_one(self, x):
        # traverse the tree until a leaf node is reached
        # while traversing go to the child node that corresponds to the value of the split feature

        node = self._tree
        while not node['leaf']:
            split_feature = node['split_feature']
            val = x[split_feature]
            try:
                node = node['children'][val]

            # occurs when there is a combination of feature values that the model has not seen before
            # in this case, return the most common label of the current node
            # label field of the non-leaf are used here
            except KeyError:                    
                return node['label']

        return node['label']       
            

    def cross_validation(self, X, y, params, cross_val_splits=5):
        # assign defaults to parameters if they are not provided
        if not params['max_depth']:
            params['max_depth'] = [None]
        if not params['min_samples_split']:
            params['min_samples_split'] = [2]

        # check if the parameters are valid
        for key in params:
            if key not in ['max_depth', 'min_samples_split']:
                raise ValueError(f"Invalid parameter: {key}")
            
        best_accuracy = 0
        len_val = len(X) // cross_val_splits

        # iterate over all the hyperparameters
        for max_depth in params['max_depth']:
            for min_samples_split in params['min_samples_split']:
                sum_accuracy = 0
                self.max_depth = max_depth if max_depth is not None else float('inf')
                self.min_samples_split = min_samples_split
                
                # iterate over all the cross validation splits
                for i in range(cross_val_splits):
                    # create the training and validation sets
                    X_train_cv = pd.concat([X.iloc[:i*len_val, :].copy(), X.iloc[(i+1)*len_val:, :].copy()])
                    X_val = X.iloc[i * len_val: (i + 1) * len_val]
                    y_train_cv = pd.concat([y_train.iloc[:i*len_val].copy(), y_train.iloc[(i+1)*len_val:].copy()])
                    y_val = y.iloc[i * len(y) // cross_val_splits: (i + 1) * len(y) // cross_val_splits]

                    # fit the model and get the accuracy on the validation set
                    self.fit(X_train_cv, y_train_cv)
                    y_pred_cv = self.predict(X_val)
                    sum_accuracy += accuracy_score(y_val, y_pred_cv)

                print(f"max_depth: {max_depth}, min_samples_split: {min_samples_split}, accuracy: {sum_accuracy / cross_val_splits}")

                # update the best hyperparameters if the current model is better
                if sum_accuracy / cross_val_splits > best_accuracy:
                    best_accuracy = sum_accuracy / cross_val_splits
                    best_params = (max_depth, min_samples_split)

        print(f"Best accuracy: {best_accuracy}, Best params: {best_params}")
        return best_params


    def _entropy(self, y):
        # calculate the entropy of a node for a given variable

        classes = np.unique(y)
        entropy = 0
        for class_ in classes:
            nominator = (y == class_).sum()
            denominator = len(y)
            item = nominator / denominator
            entropy -= item * np.log2(item)
        return entropy


    def _information_gain(self, current_entropy, old_y, new_y_list = []):
        # calculate the information gain for a given entropy, variable and list of new splits

        for y in new_y_list:
            current_entropy -= (len(y) / len(old_y)) * self._entropy(y)
        return current_entropy


    def _split_feature(self, X, y, feature):
        # split the dataset based on the unique values of a feature
        # return the new X's and y's as lists
        X_list = []
        y_list = []
        for class_ in np.unique(X[feature]):
            mask = X[feature] == class_
            X_list.append(X[mask].drop(columns=[feature]))  # drop the feature column to avoid using it again
            y_list.append(y[mask])
        return X_list, y_list


if __name__ == "__main__":
    df = pd.read_csv('loan_data.csv')

    # convert the numerical values to categorical values
    # apply this to the test data as well, so that the test data has the same categories as the training data
    numerical_classes = [
        'person_age', 
        'person_income', 
        'person_emp_exp', 
        'loan_amnt',
        'loan_int_rate', 
        'loan_percent_income', 
        'cb_person_cred_hist_length',
        'credit_score',
    ]
    for class_ in numerical_classes:
        df[class_] = pd.qcut(df[class_], 5, duplicates='drop')

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # create a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # apply cross validation to find the best hyperparameters
    dt = DecisionTree()
    params = {
        'max_depth': [2, 5, 7, 10, 12, None],
        'min_samples_split': [200, 100, 50, 20, 10, 5, 2]
    }
    best_params = dt.cross_validation(X_train, y_train, params)

    # create one more instance of the model with the best hyperparameters
    max_depth, min_samples_split = best_params
    dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
    
    # write the tree to a file to visualize it
    with open('tree.txt', 'w') as f:
        f.write(str(dt))

    # fit the model and make predictions
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # calculate the accuracy
    print(accuracy_score(y_test, y_pred))
    



"""
person_age
[ 20.  21.  22.  23.  24.  25.  26.  27.  28.  29.  30.  31.  32.  33.
  34.  35.  36.  37.  38.  39.  40.  41.  42.  43.  44.  45.  46.  47.
  48.  49.  50.  51.  52.  53.  54.  55.  56.  57.  58.  59.  60.  61.
  62.  63.  64.  65.  66.  67.  69.  70.  73.  76.  78.  80.  84.  94.
 109. 116. 123. 144.]

 person_gender
['female' 'male']

person_education
['Associate' 'Bachelor' 'Doctorate' 'High School' 'Master']

person_income
[   8000.    8037.    8104. ... 5545545. 5556399. 7200766.]

person_emp_exp
[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  57  58  61
  62  76  85  93 100 101 121 124 125]

person_home_ownership
['MORTGAGE' 'OTHER' 'OWN' 'RENT']

loan_amnt
[  500.   563.   700. ... 34800. 34826. 35000.]

loan_intent
['DEBTCONSOLIDATION' 'EDUCATION' 'HOMEIMPROVEMENT' 'MEDICAL' 'PERSONAL'
 'VENTURE']

loan_int_rate
[ 5.42  5.43  5.44 ... 19.9  19.91 20.  ]

loan_percent_income
[0.   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1  0.11 0.12 0.13
 0.14 0.15 0.16 0.17 0.18 0.19 0.2  0.21 0.22 0.23 0.24 0.25 0.26 0.27
 0.28 0.29 0.3  0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4  0.41
 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5  0.51 0.52 0.53 0.54 0.55
 0.56 0.57 0.58 0.59 0.61 0.62 0.63 0.66]

 cb_person_cred_hist_length
[ 2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19.
 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30.]

credit_score
[390 418 419 420 421 430 431 434 435 437 439 440 441 443 444 445 446 447
 448 449 450 451 453 454 455 456 457 458 459 460 461 462 463 464 465 466
 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484
 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502
 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520
 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538
 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556
 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574
 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592
 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610
 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628
 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646
 647 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664
 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682
 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700
 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718
 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736
 737 738 739 740 741 742 743 744 745 746 747 748 750 751 753 754 755 756
 759 760 762 764 765 767 768 770 772 773 784 789 792 805 807 850]

previous_loan_defaults_on_file
['No' 'Yes']
"""

