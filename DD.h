#pragma once
#include <Windows.h>


typedef int (WINAPI* pDD_btn)(int btn);
typedef int (WINAPI* pDD_whl)(int whl);
typedef int (WINAPI* pDD_key)(int keycode, int flag);
typedef int (WINAPI* pDD_mov)(int x, int y);
typedef int (WINAPI* pDD_str)(const char* str);   // DLL 是窄字符
typedef int (WINAPI* pDD_todc)(int vk);
typedef int (WINAPI* pDD_movR)(int dx, int dy);


class CDD
{
public:
    CDD() : m_hModule(nullptr),
        DD_btn(nullptr), DD_whl(nullptr), DD_key(nullptr),
        DD_mov(nullptr), DD_str(nullptr), DD_todc(nullptr), DD_movR(nullptr) {
    }

    ~CDD() {
        if (m_hModule) { ::FreeLibrary(m_hModule); m_hModule = nullptr; }
    }


    int GetFunAddr(LPCWSTR dllfile);

public:
    pDD_btn   DD_btn;    // Mouse button
    pDD_whl   DD_whl;    // Mouse wheel
    pDD_key   DD_key;    // Keyboard
    pDD_mov   DD_mov;    // Mouse move abs.
    pDD_str   DD_str;    // Input visible char (char*)
    pDD_todc  DD_todc;   // VK to ddcode
    pDD_movR  DD_movR;   // Mouse move rel.

private:
    HMODULE m_hModule;
};

