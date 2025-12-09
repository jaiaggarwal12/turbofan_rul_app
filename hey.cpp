#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

Node* createNode(int val) {
    Node* temp = new Node();
    temp->data = val;
    temp->next = NULL;
    return temp;
}

void traverse(Node* head) {
    cout << "List: ";
    Node* ptr = head;
    while(ptr != NULL){
        cout << ptr->data << " ";
        ptr = ptr->next;
    }
    cout << endl;
}

Node* insertBeg(Node* head, int val){
    Node* newNode = createNode(val);
    newNode->next = head;
    return newNode;
}

Node* insertEnd(Node* head, int val){
    Node* newNode = createNode(val);

    if(head == NULL) return newNode;

    Node* ptr = head;
    while(ptr->next != NULL)
        ptr = ptr->next;

    ptr->next = newNode;
    return head;
}

Node* insertAfter(Node* head, int after, int val){
    Node* ptr = head;

    while(ptr != NULL && ptr->data != after)
        ptr = ptr->next;

    if(ptr == NULL) return head;

    Node* newNode = createNode(val);
    newNode->next = ptr->next;
    ptr->next = newNode;

    return head;
}

Node* deleteBeg(Node* head){
    if(head == NULL) return NULL;

    Node* temp = head;
    head = head->next;
    delete temp;

    return head;
}

Node* deleteEnd(Node* head){
    if(head == NULL) return NULL;

    if(head->next == NULL){
        delete head;
        return NULL;
    }

    Node* ptr = head;
    Node* prev = NULL;

    while(ptr->next != NULL){
        prev = ptr;
        ptr = ptr->next;
    }

    prev->next = NULL;
    delete ptr;
    return head;
}

Node* deleteAfter(Node* head, int after){
    Node* ptr = head;

    while(ptr != NULL && ptr->data != after)
        ptr = ptr->next;

    if(ptr == NULL || ptr->next == NULL) return head;

    Node* del = ptr->next;
    ptr->next = del->next;
    delete del;

    return head;
}

int main(){
    Node* head = NULL;

    cout << "---- INSERTIONS ----" << endl;

    head = insertBeg(head, 30);
    head = insertBeg(head, 20);
    head = insertBeg(head, 10);
    traverse(head);

    head = insertEnd(head, 40);
    head = insertEnd(head, 50);
    traverse(head);

    head = insertAfter(head, 30, 35);
    traverse(head);

    cout << "---- DELETIONS ----" << endl;

    head = deleteBeg(head);
    traverse(head);

    head = deleteEnd(head);
    traverse(head);

    head = deleteAfter(head, 30);
    traverse(head);

    return 0;
}
