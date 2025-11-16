#include<iostream>
#include<memory>
#include<cstring>
#include<stdexcept>

class MyString
{
    private:
        std::unique_ptr<char[]> data;
        size_t length;
    
    public:

        // 参数构造函数
        MyString(const char* str = "")
        {
            if (!str)
            {
                throw std::invalid_argument("Cannot create a new string with null ptr");
            }
            
            // 使用 make_unique 创建 unique_ptr
            length = strlen(str);
            data = std::make_unique<char[]>(length + 1);
            strcpy(data.get(), str);
            std::cout << "Use para constructor sucess: " << data.get() << std::endl;
        }

        // 拷贝构造函数(移动语义)
        MyString(const MyString& other) : length(other.length)
        {
            data = std::make_unique<char[]>(length + 1);
            strcpy(data.get(), other.data.get());
            std::cout << "Use copy constructor sucess: " << data.get() << std::endl;
        }

        // 移动构造函数(转移控制权后，原资源释放)
        MyString(MyString&& other) : length(other.length)
        {
            data = std::move(other.data);
            other.length = 0;
            std::cout << "Use move constructor success: " << data.get() << std::endl;
        }

        // 拷贝赋值运算符
        MyString& operator=(const MyString& other)
        {
            if (this != &other)
            {
                length = other.length;
                data = std::make_unique<char[]>(length + 1);
                strcpy(data.get(), other.data.get());
                std::cout << "use = operator" << std::endl;
            }
            return *this;
        }

        // 移动赋值运算符
        MyString& operator=(MyString&& other)
        {
            if (this != &other)
            {
                length = other.length;
                data = std::move(other.data);
                other.length = 0;
            }

            return *this;
        }

        void display() const
        {
            std::cout << "data: " << data.get() << ", length: " << length << std::endl;
        }
};



int main()
{
    MyString myString("This is for test!");
    myString.display();

    // 测试拷贝构造
    MyString copyString = myString;
    copyString.display();

    // 测试移动构造
    MyString moveString = std::move(myString);
    moveString.display();
    myString.display();  // 显示移动后的状态

    // 测试拷贝赋值
    MyString assignString("Another string");
    assignString = copyString;
    assignString.display();

    // 测试移动赋值
    MyString moveAssignString;
    moveAssignString = std::move(copyString);
    moveAssignString.display();
    copyString.display();  // 显示移动后的状态

    return 0;
}