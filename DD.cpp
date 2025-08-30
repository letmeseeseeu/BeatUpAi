#include "DD.h"

int CDD::GetFunAddr(LPCWSTR dllfile)
{
    // ·����飨���ַ���
    DWORD attr = ::GetFileAttributesW(dllfile);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        return -11; // �ļ�������
    }

    // ���� DLL�����ַ���
    m_hModule = ::LoadLibraryW(dllfile);
    if (!m_hModule) {
        return -12; // ����ʧ��
    }

    // ������������
    DD_btn = reinterpret_cast<pDD_btn>(::GetProcAddress(m_hModule, "DD_btn"));
    DD_whl = reinterpret_cast<pDD_whl>(::GetProcAddress(m_hModule, "DD_whl"));
    DD_key = reinterpret_cast<pDD_key>(::GetProcAddress(m_hModule, "DD_key"));
    DD_mov = reinterpret_cast<pDD_mov>(::GetProcAddress(m_hModule, "DD_mov"));
    DD_str = reinterpret_cast<pDD_str>(::GetProcAddress(m_hModule, "DD_str"));
    DD_todc = reinterpret_cast<pDD_todc>(::GetProcAddress(m_hModule, "DD_todc"));
    DD_movR = reinterpret_cast<pDD_movR>(::GetProcAddress(m_hModule, "DD_movR"));

    if (DD_btn && DD_whl && DD_key && DD_mov && DD_str && DD_todc && DD_movR) {
        return 1;   // �ɹ�
    }
    return -13;     // ȱ�ٵ���
}
